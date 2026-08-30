"""Base advisor module.

Defines the Recommendation dataclass and BaseAdvisor abstract class
that all advisor modules must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single confidence-scored recommendation.

    Attributes:
        label: Human-readable recommendation title.
        confidence_score: Float between 0.0 and 1.0.
        star_rating: Integer 1–5, derived from confidence_score.
        reason: Plain-English reason grounded in dataset properties.
        why_explanation: Longer explanatory text for the Why? expander.
        category: Grouping key (e.g. 'imputation', 'encoding', 'model').
        metadata: Advisor-specific payload for downstream processing.
    """

    label: str
    confidence_score: float
    reason: str
    why_explanation: str
    star_rating: int = 1
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and derive fields after initialisation."""
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        self.star_rating = max(1, min(5, round(self.confidence_score * 5)))


class BaseAdvisor(ABC):
    """Abstract base class for all SmartML Studio advisors.

    Every advisor must implement the `recommend` method, which takes
    a dataset profile dictionary and returns a ranked list of
    Recommendation objects.
    """

    @abstractmethod
    def recommend(self, profile: dict[str, Any]) -> list[Recommendation]:
        """Generate ranked recommendations based on the dataset profile.

        Args:
            profile: Dictionary produced by DatasetProfiler.compute_profile().

        Returns:
            List of Recommendation objects sorted by confidence_score descending.
        """
        ...
