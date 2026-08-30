"""Turns a SHAP global importance ranking into plain English.

The Global Interpretability tab renders a beeswarm summary plot. It encodes three
things at once -- horizontal position for impact, colour for feature value, vertical
stacking for density -- none of them labelled in a way a non-specialist reads
correctly. This narrator describes the same ranking in prose.

As with the per-prediction narrator, every number here is computed by SHAP. This layer
only rewrites the ranking as sentences.
"""

import logging

from src.llm.client import LLMClient
from src.llm.context import build_global_payload

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You explain which features a machine-learning model relies on, to \
someone who understands their data but not machine learning.

You are given each feature's mean absolute SHAP value: how much that feature moves the \
model's output on average, with direction ignored. A larger value means the model leans \
on that feature more.

Rules:
- Use only the features and numbers in the table. Never invent a feature or a number.
- Open by naming the two or three features the model leans on most, and say roughly how \
they compare to each other using the share column.
- Mention if influence is concentrated in a few features or spread across many, and say \
plainly which features contribute almost nothing.
- Direction is NOT available here. Never say a feature raises or lowers the prediction, \
only how much the model relies on it.
- Importance is not real-world cause. Say the model relies on a feature, never that the \
feature causes the outcome.
- Do not give advice about changing the model or the data.

Write two short paragraphs of plain prose. No headings, no bullet lists, no markdown."""


class GlobalNarrator:
    """Narrates a model's overall feature reliance from aggregated SHAP values."""

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
        model_name: str,
        ranked_importance: list[tuple[str, float]],
        problem_type: str = "Classification",
        n_samples: int = 0,
        output_space: str | None = None,
    ) -> str | None:
        """Return prose describing which features the model relies on, or None.

        Returns None rather than raising so the caller keeps showing the summary plot
        with no narration instead of surfacing an error.
        """
        if not ranked_importance:
            logger.info("No global importances supplied; skipping narration.")
            return None

        payload = build_global_payload(
            model_name=model_name,
            ranked_importance=ranked_importance,
            problem_type=problem_type,
            n_samples=n_samples,
            output_space=output_space,
        )
        logger.info("Requesting global narration for %s.", model_name)
        return self.client.complete(system=_SYSTEM_PROMPT, user=payload)
