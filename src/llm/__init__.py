"""LLM explanation layer.

Narrates deterministic pipeline outputs. Never makes a pipeline decision: model
selection, preprocessing strategy, confidence scores and SHAP values are all computed
by the rule-based layer and are untouched by anything here.
"""

from src.llm.advisor_narrator import AdvisorNarrator
from src.llm.client import LLMClient
from src.llm.global_narrator import GlobalNarrator
from src.llm.prediction_narrator import PredictionNarrator
from src.llm.report_narrator import ReportNarrator

__all__ = [
    "AdvisorNarrator",
    "GlobalNarrator",
    "LLMClient",
    "PredictionNarrator",
    "ReportNarrator",
]
