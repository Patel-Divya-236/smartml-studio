"""One call that runs the whole planning layer: plan, parse, validate.

This is what the API caches per session. The merge with the rule-based advisors happens
at the endpoint, per advisor, because each advisor's rules are computed there.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.advisors.base import Recommendation
from src.llm.client import LLMClient
from src.planning.planner import LLMPlanner
from src.planning.schema import Rejection
from src.planning.validator import validate_plan

logger = logging.getLogger(__name__)


@dataclass
class AdvisorPlan:
    """The validated outcome of planning, ready to merge with the rules."""

    status: str  # "ok" | "unavailable" | "failed"
    model: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    preprocessing: list[Recommendation] = field(default_factory=list)
    feature_engineering: list[Recommendation] = field(default_factory=list)
    models: list[Recommendation] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    excluded_models: set[str] = field(default_factory=set)

    def info(self, section: str | None = None) -> dict[str, Any]:
        """What the UI shows about the planner: whether it ran, and what it refused."""
        rejected = [r for r in self.rejected if section is None or r.section == section]
        return {
            "status": self.status,
            "model": self.model,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "accepted": {
                "preprocessing": len(self.preprocessing),
                "feature_engineering": len(self.feature_engineering),
                "models": len(self.models),
            },
            "rejected": [r.to_dict() for r in rejected],
        }


def build_advisor_plan(
    profile: dict[str, Any],
    target: Any,
    client: LLMClient | None = None,
) -> AdvisorPlan:
    """Plan, parse and validate. Never raises; every failure is a status."""
    result = LLMPlanner(client).plan(profile, target)
    if result.status != "ok" or result.plan is None:
        logger.info("AI planning %s: %s", result.status, result.error)
        return AdvisorPlan(
            status=result.status,
            model=result.model,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
            rejected=list(result.rejected),
        )

    validated = validate_plan(result.plan, profile, target)
    return AdvisorPlan(
        status="ok",
        model=result.model,
        elapsed_seconds=result.elapsed_seconds,
        preprocessing=validated.preprocessing,
        feature_engineering=validated.feature_engineering,
        models=validated.models,
        rejected=list(result.rejected) + validated.rejected,
        excluded_models=validated.excluded_models,
    )
