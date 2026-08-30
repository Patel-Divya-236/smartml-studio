"""The three rule-based advisors, plus their LLM explanation layer.

The advisors decide; the language model only explains. That separation is the point of
the architecture and is enforced here by keeping the recommendation endpoints entirely
free of LLM calls — narration lives on its own routes, and a failure there returns 200
with `narrative: null` so the recommendation is never withheld because a provider is down.
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


def _recommend(session: Session, kind: str) -> list[dict]:
    """Run (or reuse) one advisor's recommendations for the current profile."""
    profile = require(session, "profile", "analysis")
    cache_key = _CACHE_KEYS[kind]

    cached = session.get(cache_key)
    if cached is None:
        cached = _ADVISORS[kind]().recommend(profile)
        session.set(cache_key, cached)
        logger.info("%s advisor produced %d recommendations.", kind, len(cached))

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

    return {"recommendations": recs, "by_column": by_column}


@router.get("/model")
def model(session: Session = Depends(get_session)) -> dict:
    """Model recommendations for the detected task type."""
    return {"recommendations": _recommend(session, "model")}


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
