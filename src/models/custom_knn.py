"""Custom kNN wrapper using sklearn.neighbors.KNeighborsClassifier.
"""

import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class CustomKNN:
    """kNN classifier wrapper supporting sklearn-compatible interface.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean") -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)
        self.classes_ = None
        logger.info("CustomKNN initialised (k=%d, metric=%s).", n_neighbors, metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomKNN":
        """Fit the kNN model."""
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        logger.info("CustomKNN fitted with sklearn back-end.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

