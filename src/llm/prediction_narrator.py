"""Turns SHAP attributions into a plain-English explanation of one prediction.

The numbers are computed by SHAP from the trained model; this only rewrites them as
prose. It never evaluates the model, never judges whether the prediction is correct,
and never introduces a feature that is not in the attribution table.
"""

import logging
from typing import Any

import numpy as np

from src.llm.client import LLMClient
from src.llm.context import build_prediction_payload

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You explain a single machine-learning prediction to someone who \
understands their data but not machine learning.

You are given SHAP attributions: each feature's contribution to moving this prediction \
away from the model's baseline. Positive pushes the prediction up, negative pushes it down.

Rules:
- Use only the features, values and contributions in the table. Never invent a feature, a value, or a cause that is not listed.
- Name the three or four largest drivers and say which direction each pushed and roughly how strongly relative to the others.
- UNITS: the payload states the contribution units. Only call a contribution a percentage or "percentage points" when the units are stated as probability. Otherwise compare contributions to each other ("about twice the effect of the next feature") and never convert them to a percentage or a probability.
- Never calculate. The payload precomputes each contribution's size relative to the largest; quote that column rather than working out ratios yourself.
- Do not claim the prediction is right or wrong, and do not give advice.
- Say "the model" - never imply the features cause the outcome in the real world, only that they drove this model's output.

Write two short paragraphs of plain prose. No headings, no bullet lists, no markdown."""


class PredictionNarrator:
    """Narrates one prediction from its SHAP attributions."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialise with an LLM client, creating a default one if not supplied."""
        self.client = client or LLMClient()

    @property
    def is_available(self) -> bool:
        """True when narration can be attempted."""
        return self.client.is_available

    def narrate(
        self,
        *,
        feature_names: list[str],
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        predicted_label: Any,
        actual_label: Any = None,
        problem_type: str = "Classification",
        base_value: float | None = None,
        output_space: str | None = None,
    ) -> str | None:
        """Return a prose explanation of this prediction, or None if unavailable.

        Returns None rather than raising so the caller keeps showing the SHAP plot
        with no narration instead of surfacing an error.
        """
        payload = build_prediction_payload(
            feature_names=feature_names,
            feature_values=feature_values,
            shap_values=shap_values,
            predicted_label=predicted_label,
            actual_label=actual_label,
            problem_type=problem_type,
            base_value=base_value,
            output_space=output_space,
        )
        logger.info("Requesting prediction narration (%d features supplied).", len(feature_names))
        return self.client.complete(system=_SYSTEM_PROMPT, user=payload)
