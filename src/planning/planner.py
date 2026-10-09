"""Asks the LLM for a preprocessing, feature-engineering and model plan.

The planner proposes; it does not decide. Its reply is parsed against `schema.py` and
then checked line by line by `validator.py` before anything reaches the user, and the
rule-based advisors fill whatever it leaves out. So a planner that is unavailable, slow,
or wrong costs the user the AI suggestions -- never the pipeline.

Like `LLMClient`, it never raises. Every outcome is a `PlanResult` with a status the UI
can explain: "ok", "unavailable" (no key configured) or "failed" (network, rate limit,
timeout, unparseable reply).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import SETTINGS
from src.llm.client import LLMClient
from src.planning.profile_digest import build_planning_profile
from src.planning.schema import PlanResponse, Rejection, extract_json, parse_plan
from src.planning.vocabulary import (
    ENCODING,
    FEATURE_ENGINEERING,
    IMPUTATION,
    SCALING,
    models_for,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """The outcome of one planning call."""

    status: str  # "ok" | "unavailable" | "failed"
    plan: PlanResponse | None = None
    rejected: list[Rejection] = field(default_factory=list)
    error: str | None = None
    model: str | None = None
    elapsed_seconds: float = 0.0


def build_system_prompt(problem_type: str | None, target: str | None) -> str:
    """The planning instructions, with the allowed vocabulary written in.

    The vocabulary and the limits are generated from the same tables and settings the
    validator enforces, so the model is told the actual rules rather than a paraphrase
    of them -- and anything it does despite them is still caught downstream.
    """
    task = problem_type or "Classification"
    models = ", ".join(models_for(problem_type))
    target_name = target if target is not None else "(none)"

    return f"""You are the planning component of an AutoML tool. You receive a JSON \
profile of a tabular dataset -- aggregate statistics only, never rows -- and return a \
preprocessing, feature-engineering and model plan for a {task} task.

Return ONE JSON object with exactly this shape and nothing else:
{{
  "preprocessing": [
    {{"column": "<name exactly as in the profile>", "step": "imputation|encoding|scaling",
      "technique": "<allowed technique>", "reason": "<one sentence>", "confidence": 0.0}}
  ],
  "feature_engineering": [
    {{"technique": "<allowed technique>", "params": {{}}, "reason": "<one sentence>", "confidence": 0.0}}
  ],
  "models": [
    {{"model": "<allowed model>", "reason": "<one sentence>", "confidence": 0.0}}
  ]
}}

Allowed techniques:
- imputation: {", ".join(IMPUTATION)}. Only for columns with missing_pct > 0, except \
"drop", which may be used on any column. mean, median and knn need a numeric column.
- encoding: {", ".join(ENCODING)}. Only for kind "categorical". onehot only when distinct \
<= {SETTINGS.ONEHOT_MAX_CARDINALITY}; use ordinal above that.
- scaling: {", ".join(SCALING)}. Only for kind "numeric". log1p only when min > -1.
- feature_engineering: {", ".join(FEATURE_ENGINEERING)}, with params:
  low_variance {{"threshold": number between 0 and 1}};
  polynomial {{"degree": 2 or 3, "interaction_only": true|false}} -- only with at most \
{SETTINGS.POLY_MAX_NUMERIC} numeric features;
  pca {{"n_components": integer, at least 1 and at most the number of numeric features}};
  select_k_best {{"k": integer, fewer than the number of features}}.
- models: {models}.

Rules:
- Never include the target column "{target_name}" in preprocessing.
- Columns of kind "high_cardinality_text" are identifiers, dates or free text: drop them.
- At most one technique per column per step. A dropped column gets no other steps.
- Every column with missing_pct > 0 that you keep needs an imputation step, and every \
categorical column you keep needs an encoding step.
- Feature engineering is optional. Recommend a technique only when the profile gives a \
concrete reason (for example strongly correlated features for pca); an empty list is a \
valid answer.
- Recommend 3 to 6 models, best first. SVM is slow on large data: avoid it above \
{SETTINGS.SVM_MAX_ROWS:,} rows.
- Each reason must cite the profile's own numbers (missing %, skew, distinct values, \
rows, imbalance ratio, correlation). Never invent a number.
- confidence is your certainty from 0.0 to 1.0. Use the full range.
- Output JSON only: no markdown, no comments, no text outside the object."""


class LLMPlanner:
    """Produces a parsed (not yet validated) plan from a dataset profile."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    @property
    def is_available(self) -> bool:
        return self.client.is_available

    def plan(self, profile: dict[str, Any], target: str | None) -> PlanResult:
        """Request a plan. Returns a PlanResult whatever happens; never raises."""
        model = getattr(self.client, "model", None)
        if not self.client.is_available:
            return PlanResult(
                status="unavailable",
                error=self.client.last_error or "No LLM_API_KEY configured.",
                model=model,
            )

        try:
            digest = build_planning_profile(profile, target)
            system = build_system_prompt(profile.get("problem_type"), target)
            user = json.dumps(digest, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:  # a profile shape this code did not anticipate
            logger.exception("Could not build the planning payload")
            return PlanResult(status="failed", error=f"Could not describe the dataset: {exc}", model=model)

        extra: dict[str, Any] = {}
        if SETTINGS.LLM_PLAN_REASONING_EFFORT:
            extra["reasoning_effort"] = SETTINGS.LLM_PLAN_REASONING_EFFORT

        logger.info("Requesting an AI plan (%d columns described).", len(digest.get("columns", [])))
        started = time.perf_counter()
        text = self.client.complete(
            system=system,
            user=user,
            max_tokens=SETTINGS.LLM_PLAN_MAX_TOKENS,
            temperature=SETTINGS.LLM_PLAN_TEMPERATURE,
            response_format={"type": "json_object"},
            extra=extra,
            timeout=SETTINGS.LLM_PLAN_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started

        if text is None:
            return PlanResult(
                status="failed",
                error=self.client.last_error or "The planner returned nothing.",
                model=model,
                elapsed_seconds=elapsed,
            )

        raw = extract_json(text)
        if raw is None:
            logger.warning("Planner reply was not a JSON object: %r", text[:200])
            return PlanResult(
                status="failed",
                error="The planner's reply was not valid JSON.",
                model=model,
                elapsed_seconds=elapsed,
            )

        plan, rejected = parse_plan(raw)
        logger.info(
            "AI plan received in %.1fs: %d preprocessing, %d feature, %d model item(s); %d malformed.",
            elapsed, len(plan.preprocessing), len(plan.feature_engineering), len(plan.models), len(rejected),
        )
        return PlanResult(
            status="ok",
            plan=plan,
            rejected=rejected,
            model=model,
            elapsed_seconds=elapsed,
        )
