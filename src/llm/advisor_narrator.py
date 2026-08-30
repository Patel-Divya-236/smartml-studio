"""Expands a rule-based advisor recommendation into beginner-facing English.

This is the layer a first-time user needs on the Preprocessing, Feature Engineering and
Model Advisor pages: not just *what* was recommended, but what the technique actually
does and why this particular column triggered it.

The recommendation itself is produced entirely by the rule-based advisors. Confidence
scores, thresholds and the choice of action are computed in `src/advisors/` and are not
open to revision here -- the model is handed a decision and asked to explain it. That
separation is deliberate: the confidence-scored advisor layer stays the system's own
contribution rather than becoming a wrapper around a chat model.
"""

import logging
from typing import Any

from src.advisors.base import Recommendation
from src.llm.client import LLMClient
from src.llm.context import build_column_payload, build_recommendation_payload

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You explain a data-preparation or modelling recommendation to \
someone who knows their own data but has no machine-learning background.

A rule-based advisor has already made this recommendation. Your job is to explain it, \
not to second-guess it.

Cover exactly two things, in this order:
1. What the recommended technique actually does, in everyday terms. Assume the reader \
has never heard of it. One plain sentence of mechanism is worth more than jargon.
2. Why it was recommended for this specific column or dataset, pointing at the actual \
numbers given (missing percentage, distinct value count, skewness, row count).

Rules:
- Use only the facts supplied. Never invent a statistic, and never quote a number that \
is not in the payload.
- Never contradict the recommendation, suggest a different action, or say the rule is \
wrong. If a genuine trade-off exists, state it as something to be aware of, not as a \
reason to do something else.
- Define any technical term the moment you use it.
- No headings, no bullet lists, no markdown.

Write one short paragraph for each of the two points. Be concrete and calm."""


class AdvisorNarrator:
    """Explains one advisor recommendation in beginner-facing prose."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialise with an LLM client, creating a default one if not supplied."""
        self.client = client or LLMClient()

    @property
    def is_available(self) -> bool:
        """True when narration can be attempted."""
        return self.client.is_available

    def explain(
        self,
        recommendation: Recommendation,
        profile: dict[str, Any] | None = None,
    ) -> str | None:
        """Return a plain-English expansion of one recommendation, or None.

        Returns None rather than raising so the page falls back to the advisor's own
        static `why_explanation` text instead of surfacing an error.
        """
        payload = build_recommendation_payload(
            label=recommendation.label,
            category=recommendation.category,
            reason=recommendation.reason,
            why_explanation=recommendation.why_explanation,
            confidence_score=recommendation.confidence_score,
            metadata=recommendation.metadata,
            profile=profile,
        )
        logger.info("Requesting advisor narration for %r.", recommendation.label)
        return self.client.complete(system=_SYSTEM_PROMPT, user=payload)

    def explain_column(
        self,
        column: str,
        recommendations: list[Recommendation],
        profile: dict[str, Any] | None = None,
    ) -> str | None:
        """Explain every preparation step recommended for one column, together.

        Returns None rather than raising, so the page falls back to the advisors' own
        static text.
        """
        if not recommendations:
            return None

        payload = build_column_payload(
            column=column,
            recommendations=recommendations,
            profile=profile,
        )
        logger.info("Requesting column narration for %r (%d steps).", column, len(recommendations))
        return self.client.complete(system=_SYSTEM_PROMPT, user=payload)
