"""The deterministic gate between what the planner proposes and what a user sees.

The planner decides what is *appropriate* for a dataset; this module decides whether each
proposal is *valid and safe to execute*. Every check here is a fixed rule over the
profile -- no model call, no randomness -- so the same plan against the same dataset is
always judged the same way, and every refusal comes with a reason a user can read.

What gets rejected, and why it matters:

* Names that do not exist: an unknown column, technique or model is not an unusual choice,
  it is an action no code path can run.
* Type errors: scaling a text column, averaging a category, encoding a number. The
  pipeline would raise mid-run, far from the choice that caused it.
* Values the transform cannot take: log1p at or below -1 produces NaN.
* Choices that are valid but unsafe on a small host: one-hot encoding hundreds of
  categories, polynomial expansion into hundreds of columns, SVM on tens of thousands of
  rows. Each of those has either exhausted a 512MB instance or run for minutes.
* Contradictions within the plan: two scalers for one column, scaling a column the same
  plan drops, the prediction target transformed like a feature.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from config.settings import SETTINGS
from src.advisors.base import Recommendation
from src.planning.profile_digest import column_kind
from src.planning.schema import ColumnStep, FeatureStep, ModelPick, PlanResponse, Rejection
from src.planning.vocabulary import (
    FEATURE_ENGINEERING,
    FEATURE_EXPLANATIONS,
    MODEL_EXPLANATIONS,
    NUMERIC_ONLY_IMPUTERS,
    PREPROCESSING_CATEGORY,
    PREPROCESSING_LABELS,
    TECHNIQUE_EXPLANATIONS,
    model_label,
    models_for,
    normalise_model,
    normalise_technique,
)

logger = logging.getLogger(__name__)

SOURCE = "llm"

_SCALER_NAMES = {
    "standard": "standard scaling",
    "minmax": "min-max scaling",
    "robust": "robust scaling",
    "log1p": "a log transform",
}


@dataclass
class ValidatedPlan:
    """Recommendations that passed every check, and every proposal that did not."""

    preprocessing: list[Recommendation] = field(default_factory=list)
    feature_engineering: list[Recommendation] = field(default_factory=list)
    models: list[Recommendation] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    # Models refused on safety grounds rather than for a bad name. The rule ranking must
    # not quietly add these back when it tops up a short model list.
    excluded_models: set[str] = field(default_factory=set)


def _trim(reason: str) -> str:
    limit = SETTINGS.PLAN_MAX_REASON_CHARS
    return reason if len(reason) <= limit else reason[: limit - 1].rstrip() + "…"


def _whole_number(value: Any) -> int | None:
    """An integer from 5, 5.0 or "5"; None for 0.95, True, or anything else."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _resolve_column(name: str, columns: dict[str, Any]) -> Any | None:
    """Find the profile's column for a name, tolerating a change of letter case only."""
    if name in columns:
        return columns[name]
    folded = [original for text, original in columns.items() if text.lower() == name.lower()]
    return folded[0] if len(folded) == 1 else None


# ── Preprocessing ───────────────────────────────────────────────────────

def _preprocessing_problem(
    technique: str,
    column: Any,
    kind: str,
    missing_pct: float,
    distinct: int,
    minimum: float | None,
) -> str | None:
    """Why this technique cannot be applied to this column, or None when it can."""
    if technique == "drop":
        return None

    if kind == "high_cardinality_text":
        return (
            f"'{column}' has {distinct:,} distinct values -- an identifier, date or free "
            "text rather than a category. Only dropping it is accepted: encoding it gives "
            "almost every row its own level, and filling or scaling it is meaningless."
        )

    category = PREPROCESSING_CATEGORY[technique]
    if category == "imputation":
        if not missing_pct or missing_pct <= 0:
            return f"'{column}' has no missing values to fill."
        if technique in NUMERIC_ONLY_IMPUTERS and kind != "numeric":
            return (
                f"'{column}' holds categories, and {technique} imputation computes from "
                "numbers. Mode imputation is the one that works for categories."
            )

    elif category == "encoding":
        if kind == "numeric":
            return f"'{column}' is already numeric; encoding applies to text categories."
        if technique == "onehot" and distinct > SETTINGS.ONEHOT_MAX_CARDINALITY:
            return (
                f"'{column}' has {distinct} categories, so one-hot encoding would add "
                f"{distinct} columns (the limit is {SETTINGS.ONEHOT_MAX_CARDINALITY}). "
                "Ordinal encoding keeps it to one."
            )

    elif category == "scaling":
        if kind != "numeric":
            return (
                f"'{column}' is a text column, and {_SCALER_NAMES[technique]} needs "
                "numbers. It has to be encoded, not scaled."
            )
        if technique == "log1p" and minimum is not None and minimum <= -1:
            return (
                f"'{column}' has a minimum of {minimum:g}; log1p is undefined at or below "
                "-1 and would turn those values into missing ones."
            )

    return None


def _validate_preprocessing(
    steps: list[ColumnStep],
    profile: dict[str, Any],
    target: Any,
    rejected: list[Rejection],
) -> list[Recommendation]:
    columns = {str(c): c for c in (profile.get("dtypes") or {})}
    missing = profile.get("missing_pct") or {}
    cardinality = profile.get("cardinality") or {}
    minimums = profile.get("numeric_min") or {}

    def reject(item: str, reason: str) -> None:
        rejected.append(Rejection("preprocessing", item, reason))

    # First pass: everything that is valid on its own.
    candidates: list[tuple[Any, str, ColumnStep]] = []
    for step in steps:
        technique = normalise_technique(step.technique)
        column = _resolve_column(step.column, columns)
        item = f"{technique or step.technique} on '{column if column is not None else step.column}'"

        if technique not in PREPROCESSING_CATEGORY:
            reject(item, f"'{step.technique}' is not a preprocessing technique the pipeline can run.")
            continue
        if column is None:
            reject(item, f"There is no column named '{step.column}' in this dataset.")
            continue
        if target is not None and str(column) == str(target):
            reject(item, f"'{column}' is the prediction target. Transforming it would change "
                         "what the model predicts rather than how it learns.")
            continue
        if not step.reason:
            reject(item, "It gives no reason, and every recommendation must explain itself.")
            continue

        problem = _preprocessing_problem(
            technique,
            column,
            column_kind(column, profile),
            float(missing.get(column, 0.0) or 0.0),
            int(cardinality.get(column, 0) or 0),
            minimums.get(column),
        )
        if problem:
            reject(item, problem)
            continue
        candidates.append((column, technique, step))

    # Second pass: consistency within the plan. A drop settles the column, so any other
    # step for it is moot -- and one technique per column per step.
    dropped = {column for column, technique, _ in candidates if technique == "drop"}
    seen: set[tuple[Any, str]] = set()
    accepted: list[Recommendation] = []
    for column, technique, step in candidates:
        category = PREPROCESSING_CATEGORY[technique]
        item = f"{technique} on '{column}'"
        if column in dropped and technique != "drop":
            reject(item, f"The same plan drops '{column}', so this step would never run.")
            continue
        if (column, category) in seen:
            reject(item, f"'{column}' already has a {category} step in this plan; only the first is kept.")
            continue
        seen.add((column, category))
        accepted.append(
            Recommendation(
                label=PREPROCESSING_LABELS[technique].format(column=column),
                confidence_score=step.confidence,
                reason=_trim(step.reason),
                why_explanation=TECHNIQUE_EXPLANATIONS[technique],
                category=category,
                metadata={"column": column, "action": technique, "source": SOURCE},
            )
        )
    return accepted


# ── Feature engineering ─────────────────────────────────────────────────

def _polynomial_outputs(inputs: int, degree: int, interaction_only: bool) -> int:
    """Columns PolynomialFeatures(include_bias=False) produces from `inputs` columns."""
    if interaction_only:
        return sum(math.comb(inputs, d) for d in range(1, min(degree, inputs) + 1))
    return math.comb(inputs + degree, degree) - 1


def _feature_config(
    technique: str,
    params: dict[str, Any],
    features: int,
) -> tuple[dict[str, Any], str, str | None]:
    """Return (FeatureRequest fields, card title, problem or None).

    `features` is how many columns reach feature engineering. It runs on the *encoded*
    frame, where every kept category has become at least one numeric column, so the count
    is numeric plus categorical features -- a lower bound, since one-hot adds more.
    """
    if technique == "low_variance":
        try:
            threshold = float(params.get("threshold", SETTINGS.LOW_VARIANCE_THRESHOLD))
        except (TypeError, ValueError):
            threshold = float("nan")
        if not (0.0 < threshold < 1.0):
            return {}, "", "The variance threshold must be a number between 0 and 1."
        return (
            {"low_variance_active": True, "low_variance_threshold": threshold},
            f"Low-Variance Filter (threshold {threshold:g})",
            None,
        )

    if technique == "polynomial":
        degree = _whole_number(params.get("degree", 2))
        interaction_only = bool(params.get("interaction_only", False))
        if degree not in (2, 3):
            return {}, "", "Polynomial degree must be 2 or 3."
        if features < 1:
            return {}, "", "There are no numeric features to expand."
        outputs = _polynomial_outputs(features, degree, interaction_only)
        limit = _polynomial_outputs(SETTINGS.POLY_MAX_NUMERIC, 2, False)
        if outputs > limit:
            return {}, "", (
                f"Degree {degree} on at least {features} features after encoding makes "
                f"{outputs:,} columns (the limit is {limit}). That many columns multiplies "
                "memory and training time for every model."
            )
        suffix = ", interactions only" if interaction_only else ""
        return (
            {"poly_active": True, "poly_degree": degree, "poly_interaction_only": interaction_only},
            f"Polynomial Features (degree {degree}{suffix})",
            None,
        )

    if technique == "pca":
        if features < 2:
            return {}, "", "PCA needs at least two features to combine."
        components = _whole_number(params.get("n_components"))
        if components is None:
            return {}, "", ("PCA needs a whole number of components. A fraction of variance "
                            "is not something this pipeline accepts.")
        if not (1 <= components < features):
            return {}, "", (f"n_components must be between 1 and {features - 1}: there are "
                            f"{features} features, and keeping all of them reduces nothing.")
        return {"pca_active": True, "pca_components": components}, f"PCA ({components} components)", None

    if technique == "select_k_best":
        k = _whole_number(params.get("k"))
        if k is None:
            return {}, "", "SelectKBest needs a whole number k of features to keep."
        if not (1 <= k < features):
            return {}, "", (f"k must be between 1 and {features - 1}: there are {features} "
                            "features, and keeping all of them selects nothing.")
        return {"select_k_best_active": True, "select_k_best_k": k}, f"SelectKBest (k = {k})", None

    return {}, "", "Not a feature-engineering technique the pipeline can run."


def _validate_features(
    steps: list[FeatureStep],
    profile: dict[str, Any],
    target: Any,
    dropped: set[Any],
    rejected: list[Rejection],
) -> list[Recommendation]:
    kept = [c for c in (profile.get("dtypes") or {}) if c != target and c not in dropped]
    features = sum(column_kind(c, profile) in ("numeric", "categorical") for c in kept)

    seen: set[str] = set()
    accepted: list[Recommendation] = []
    for step in steps:
        technique = normalise_technique(step.technique)
        item = technique or step.technique

        if technique not in FEATURE_ENGINEERING:
            rejected.append(Rejection("feature_engineering", item,
                                      f"'{step.technique}' is not a feature-engineering technique the pipeline can run."))
            continue
        if technique in seen:
            rejected.append(Rejection("feature_engineering", item,
                                      f"{technique} is already in this plan; only the first is kept."))
            continue
        if not step.reason:
            rejected.append(Rejection("feature_engineering", item,
                                      "It gives no reason, and every recommendation must explain itself."))
            continue

        config, label, problem = _feature_config(technique, step.params, features)
        if problem:
            rejected.append(Rejection("feature_engineering", item, problem))
            continue

        seen.add(technique)
        accepted.append(
            Recommendation(
                label=label,
                confidence_score=step.confidence,
                reason=_trim(step.reason),
                why_explanation=FEATURE_EXPLANATIONS[technique],
                category="feature_engineering",
                metadata={"technique": technique, "config": config, "source": SOURCE},
            )
        )
    return accepted


# ── Models ──────────────────────────────────────────────────────────────

def _validate_models(
    picks: list[ModelPick],
    profile: dict[str, Any],
    rejected: list[Rejection],
    excluded: set[str],
) -> list[Recommendation]:
    task = profile.get("problem_type") or "Classification"
    allowed = models_for(task)
    rows = int((profile.get("shape") or (0, 0))[0] or 0)

    seen: set[str] = set()
    accepted: list[Recommendation] = []
    for pick in picks:
        name = normalise_model(pick.model, allowed)
        if name is None:
            rejected.append(Rejection("models", pick.model,
                                      f"'{pick.model}' is not available for a {task} task here. "
                                      f"The trainer can build: {', '.join(allowed)}."))
            continue
        if name in seen:
            rejected.append(Rejection("models", name, f"{name} is already in this plan."))
            continue
        if not pick.reason:
            rejected.append(Rejection("models", name,
                                      "It gives no reason, and every recommendation must explain itself."))
            continue
        if name == "SVM" and rows > SETTINGS.SVM_MAX_ROWS:
            excluded.add(name)
            rejected.append(Rejection("models", name,
                                      f"{rows:,} rows is above the {SETTINGS.SVM_MAX_ROWS:,}-row limit "
                                      "for SVM, whose training time grows with the square of the rows "
                                      "(at 50,000 rows it takes longer than every other model combined). It "
                                      "can still be ticked by hand on the training step."))
            continue

        seen.add(name)
        accepted.append(
            Recommendation(
                label=model_label(name, task),
                confidence_score=pick.confidence,
                reason=_trim(pick.reason),
                why_explanation=MODEL_EXPLANATIONS.get(name, ""),
                category="Classification" if task == "Classification" else "Regression",
                metadata={"model_name": name, "source": SOURCE},
            )
        )
    return accepted


# ── Entry point ─────────────────────────────────────────────────────────

def validate_plan(plan: PlanResponse, profile: dict[str, Any], target: Any) -> ValidatedPlan:
    """Check every proposal against the dataset. Never raises on a bad proposal."""
    result = ValidatedPlan()
    result.preprocessing = _validate_preprocessing(plan.preprocessing, profile, target, result.rejected)

    dropped = {r.metadata["column"] for r in result.preprocessing if r.metadata.get("action") == "drop"}
    result.feature_engineering = _validate_features(
        plan.feature_engineering, profile, target, dropped, result.rejected
    )
    result.models = _validate_models(plan.models, profile, result.rejected, result.excluded_models)

    logger.info(
        "Validator accepted %d preprocessing, %d feature, %d model item(s); rejected %d.",
        len(result.preprocessing), len(result.feature_engineering), len(result.models),
        len(result.rejected),
    )
    return result
