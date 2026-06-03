"""Custom kNN implementation from scratch.

Implements k-Nearest Neighbours with configurable distance
metric and k value. No sklearn.neighbors internals are used.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class CustomKNN:
    """From-scratch kNN classifier with sklearn-compatible interface.

    Supports configurable distance metrics (euclidean, manhattan)
    and k value. Exposes fit() and predict() for pipeline compatibility.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean") -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        logger.info("CustomKNN initialised (k=%d, metric=%s).", n_neighbors, metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomKNN":
        """Fit the kNN model (stores training data)."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        logger.info("CustomKNN fitted with %d samples.", len(X))
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between X and self.X_train."""
        if self.metric == "euclidean":
            # sqrt(sum((x - y)^2))
            # Vectorized computation
            dists = np.sqrt(
                np.sum(X**2, axis=1, keepdims=True)
                + np.sum(self.X_train**2, axis=1)
                - 2 * np.dot(X, self.X_train.T)
            )
            # Clip negative values resulting from float precision limits
            return np.clip(dists, 0.0, None)
        elif self.metric == "manhattan":
            # sum(|x - y|)
            # For each test point, compute absolute difference with all train points
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        X_arr = np.asarray(X)
        dists = self._compute_distances(X_arr)
        
        preds = []
        for idx in range(len(X_arr)):
            # Find k nearest indices
            k_indices = np.argsort(dists[idx])[:self.n_neighbors]
            # Retrieve labels
            k_labels = self.y_train[k_indices]
            # Vote
            unique_labels, counts = np.unique(k_labels, return_counts=True)
            winner = unique_labels[np.argmax(counts)]
            preds.append(winner)
            
        return np.array(preds)
