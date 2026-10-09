"""The advisors: an LLM planner, a deterministic validator, and rule-based fallbacks.

Preprocessing, feature-engineering and model recommendations come from the planning layer
in `src/planning/`. The language model proposes a plan for the whole dataset in one call;
a deterministic validator rejects anything invalid or unsafe to execute; the rule-based
advisors fill every gap the planner leaves. The model proposes, the validator decides, and
the user still approves or edits every choice before the pipeline runs.

The planner never withholds a recommendation. With no API key, a timeout, a rate limit or
an unparseable reply, the endpoints return the rule-based recommendations exactly as they
were before the planner existed, with `planner.status` saying why.

Visualization stays purely rule-based. Narration ("Explain in plain English") lives on its
own routes, and a failure there returns 200 with `narrative: null`.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.api.deps import get_session, require
from backend.core.serialization import recommendation_to_dict
from backend.core.session import Session
from src.advisors.base import Recommendation
from src.advisors.model_advisor import ModelAdvisor
from src.advisors.preprocessing_advisor import PreprocessingAdvisor
from src.advisors.visualization_advisor import VisualizationAdvisor
from src.llm.advisor_narrator import AdvisorNarrator
from src.planning.merge import merge_models, merge_preprocessing, tag
from src.planning.service import AdvisorPlan, build_advisor_plan

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/advisors", tags=["advisors"])

_CACHE_KEYS = {
    "visualization": "viz_recommendations",
    "preprocessing": "preprocessing_recommendations",
    "model": "model_recommendations",
}
_ADVISORS = {
    "visualization": VisualizationAdvisor,
    "preprocessing": PreprocessingAdvisor,
    "model": ModelAdvisor,
}


class ExplainRequest(BaseModel):
    """One recommendation to expand into beginner-facing prose."""

    label: str
    category: str = ""
    reason: str = ""
    why_explanation: str = ""
    confidence_score: float = 0.5
    metadata: dict = {}


class ExplainColumnRequest(BaseModel):
    """Every recommendation that applies to one column."""

    column: str
    recommendations: list[ExplainRequest]


def _plan(session: Session) -> AdvisorPlan:
    """The session's AI plan, computed on first use and cached.

    One planning call covers preprocessing, feature engineering and models together, so
    whichever advisor page is opened first pays for it and the others read the cache. The
    lock makes concurrent first requests wait for that one call instead of each making
    their own. The result is checkpointed: it is the only advisor output that costs a
    network round trip to recompute, so a restart should not repeat it.
    """
    cached = session.get("ai_plan")
    if cached is not None:
        return cached

    with session.plan_lock:
        cached = session.get("ai_plan")
        if cached is not None:
            return cached
        profile = require(session, "profile", "analysis")
        plan = build_advisor_plan(profile, session.get("target_column"))
        session.set("ai_plan", plan)
        session.checkpoint()
        return plan


def _recommend(session: Session, kind: str) -> list[dict]:
    """Run (or reuse) one advisor's recommendations for the current profile.

    Preprocessing and model recommendations are the AI plan merged over the rule-based
    advisor; visualization is the rule-based advisor alone.
    """
    profile = require(session, "profile", "analysis")
    cache_key = _CACHE_KEYS[kind]

    cached = session.get(cache_key)
    if cached is None:
        rules = tag(_ADVISORS[kind]().recommend(profile), "rule")
        if kind == "preprocessing":
            cached = merge_preprocessing(_plan(session).preprocessing, rules, profile)
        elif kind == "model":
            plan = _plan(session)
            cached = merge_models(plan.models, rules, plan.excluded_models)
        else:
            cached = rules
        session.set(cache_key, cached)
        logger.info(
            "%s advisor produced %d recommendations (%d from the AI plan).",
            kind, len(cached), sum(r.metadata.get("source") == "llm" for r in cached),
        )

    return [recommendation_to_dict(r) for r in cached]


@router.get("/visualization")
def visualization(session: Session = Depends(get_session)) -> dict:
    """Chart recommendations for this dataset."""
    return {"recommendations": _recommend(session, "visualization")}


@router.get("/preprocessing")
def preprocessing(session: Session = Depends(get_session)) -> dict:
    """Per-column imputation, encoding and scaling recommendations.

    Also returns them grouped by column, which is the shape the preprocessing table
    renders — one row per column with its recommended actions pre-selected.
    """
    recs = _recommend(session, "preprocessing")

    by_column: dict[str, list[dict]] = {}
    for rec in recs:
        column = rec["metadata"].get("column")
        if column:
            by_column.setdefault(str(column), []).append(rec)

    return {
        "recommendations": recs,
        "by_column": by_column,
        "planner": _plan(session).info("preprocessing"),
    }


@router.get("/model")
def model(session: Session = Depends(get_session)) -> dict:
    """Model recommendations for the detected task type."""
    recs = _recommend(session, "model")
    return {"recommendations": recs, "planner": _plan(session).info("models")}


@router.get("/features")
def features(session: Session = Depends(get_session)) -> dict:
    """Feature-engineering recommendations from the AI plan.

    There is no rule-based feature advisor: feature engineering is opt-in, and with no
    plan the step opens with every technique off -- which is a valid pipeline, not a gap.
    Each recommendation's `metadata.config` holds the exact `FeatureRequest` fields to set.
    """
    require(session, "profile", "analysis")
    plan = _plan(session)
    return {
        "recommendations": [recommendation_to_dict(r) for r in plan.feature_engineering],
        "planner": plan.info("feature_engineering"),
    }


@router.post("/explain")
def explain(payload: ExplainRequest, session: Session = Depends(get_session)) -> dict:
    """Expand one recommendation into plain English.

    Returns 200 with `narrative: null` when no model is configured or the call fails —
    the static `why_explanation` is already on screen, so an outage costs the reader
    nothing and must not surface as an error.
    """
    narrator = AdvisorNarrator()
    if not narrator.is_available:
        return {"narrative": None, "available": False, "error": narrator.client.last_error}

    recommendation = Recommendation(
        label=payload.label,
        confidence_score=payload.confidence_score,
        reason=payload.reason,
        why_explanation=payload.why_explanation,
        category=payload.category,
        metadata=payload.metadata,
    )
    narrative = narrator.explain(recommendation, session.get("profile"))
    return {
        "narrative": narrative,
        "available": True,
        "error": None if narrative else narrator.client.last_error,
    }


@router.post("/explain-column")
def explain_column(
    payload: ExplainColumnRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Explain every preparation step recommended for one column, together.

    Steps applied together are explained together, because the reason for one frequently
    depends on another.
    """
    narrator = AdvisorNarrator()
    if not narrator.is_available:
        return {"narrative": None, "available": False, "error": narrator.client.last_error}

    recommendations = [
        Recommendation(
            label=item.label,
            confidence_score=item.confidence_score,
            reason=item.reason,
            why_explanation=item.why_explanation,
            category=item.category,
            metadata=item.metadata,
        )
        for item in payload.recommendations
    ]
    narrative = narrator.explain_column(payload.column, recommendations, session.get("profile"))
    return {
        "narrative": narrative,
        "available": True,
        "error": None if narrative else narrator.client.last_error,
    }


@router.get("/llm-status")
def llm_status() -> dict:
    """Whether narration is currently available, for the UI to hide buttons cleanly."""
    narrator = AdvisorNarrator()
    return {"available": narrator.is_available, "error": narrator.client.last_error}
