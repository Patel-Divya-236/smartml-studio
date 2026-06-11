"""Hybrid Ensemble module.

Implements a custom ensemble combining user-selected models
via majority voting, weighted voting (classification), or
averaging (regression).
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """Custom hybrid ensemble combining multiple trained models.

    Supports majority voting, weighted voting (classification),
    and averaging (regression). Exposes fit/predict interface.
    """

    def __init__(self, voting: str = "majority",
                 weights: list[float] | None = None,
                 problem_type: str = "Classification") -> None:
        """Initialise the HybridEnsemble."""
        self.voting = voting
        self.weights = weights
        self.problem_type = problem_type
        self.models: list[Any] = []
        logger.info("HybridEnsemble initialised (voting=%s, problem_type=%s).", voting, problem_type)

    def fit(self, models: list[Any]) -> "HybridEnsemble":
        """Store the trained models for ensemble prediction."""
        if not models:
            raise ValueError("Must provide at least one model to create an ensemble.")
        self.models = models
        logger.info("HybridEnsemble fitted with %d base models.", len(self.models))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.models:
            raise ValueError("The ensemble has not been fitted with models yet.")

        # Gather predictions from all models
        preds = []
        for model in self.models:
            pred = model.predict(X)
            preds.append(np.array(pred).flatten())
        
        preds = np.array(preds)  # Shape: (n_models, n_samples)
        n_models, n_samples = preds.shape

        # Normalize weights if provided
        weights = self.weights
        if weights is not None:
            if len(weights) != n_models:
                logger.warning("Length of weights (%d) does not match number of models (%d). Defaulting to uniform weights.", len(weights), n_models)
                weights = [1.0] * n_models
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
        else:
            weights = np.array([1.0 / n_models] * n_models)

        if self.problem_type == "Classification":
            if self.voting == "weighted":
                # For each sample, find class with the maximum weighted vote
                final_preds = []
                for i in range(n_samples):
                    sample_preds = preds[:, i]
                    vote_counts = {}
                    for model_idx, pred_class in enumerate(sample_preds):
                        vote_counts[pred_class] = vote_counts.get(pred_class, 0.0) + weights[model_idx]
                    best_class = max(vote_counts, key=vote_counts.get)
                    final_preds.append(best_class)
                return np.array(final_preds)
            else:
                # Majority voting
                final_preds = []
                for i in range(n_samples):
                    sample_preds = preds[:, i]
                    # Compute mode
                    values, counts = np.unique(sample_preds, return_counts=True)
                    best_idx = np.argmax(counts)
                    final_preds.append(values[best_idx])
                return np.array(final_preds)
        else:
            # Regression: average or weighted average
            if self.voting == "weighted":
                return np.average(preds, axis=0, weights=weights)
            else:
                return np.mean(preds, axis=0)

