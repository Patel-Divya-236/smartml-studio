"""SmartML Studio — advisor subpackage.

Contains the recommendation engine advisors for visualization,
preprocessing, and model selection.
"""

from src.advisors.base import BaseAdvisor, Recommendation

__all__ = ["BaseAdvisor", "Recommendation"]
