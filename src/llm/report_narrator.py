"""Writes the executive summary that opens the pipeline evaluation report.

Interprets the dataset profile and the model-comparison table that the deterministic
pipeline already produced. It does not recompute metrics, rank models, or recommend
anything — the comparison table remains the source of truth.
"""

import logging
from typing import Any

import pandas as pd

from src.llm.client import LLMClient
from src.llm.context import build_report_payload

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You write the executive summary that opens a machine-learning \
pipeline report, for a reader who will read the detailed tables below it.

You are given a dataset profile and a table of trained models with their metrics.

Rules:
- Use only the figures provided. Never invent a metric, a model, or a dataset property.
- Cover: what the dataset looked like and any quality issues worth noting; which model \
performed best and on which metric; and one honest caveat about what these numbers do \
not establish (for example a small test set, class imbalance, or models that scored \
very close together).
- Quote figures to two or three significant digits.
- Be measured. Do not oversell the result.

Write three short paragraphs of plain prose. No headings, no bullet lists, no markdown."""


class ReportNarrator:
    """Narrates the pipeline outcome for the downloadable report."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """Initialise with an LLM client, creating a default one if not supplied."""
        self.client = client or LLMClient()

    @property
    def is_available(self) -> bool:
        """True when narration can be attempted."""
        return self.client.is_available

    def summarize(
        self,
        *,
        dataset_name: str,
        target_column: str,
        problem_type: str,
        profile: dict[str, Any],
        comparison: pd.DataFrame | None,
        report_state: dict[str, Any] | None = None,
    ) -> str | None:
        """Return an executive summary, or None if unavailable.

        The caller omits the summary section entirely when this returns None, so the
        report still builds without an LLM.
        """
        payload = build_report_payload(
            dataset_name=dataset_name,
            target_column=target_column,
            problem_type=problem_type,
            profile=profile,
            comparison=comparison,
            report_state=report_state,
        )
        logger.info("Requesting report executive summary.")
        return self.client.complete(system=_SYSTEM_PROMPT, user=payload)
