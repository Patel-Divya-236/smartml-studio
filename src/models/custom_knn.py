"""k-nearest-neighbours classifier and regressor implemented from scratch.

Only numpy is used: distances, neighbour selection and voting are all computed here.
This file previously wrapped `sklearn.neighbors.KNeighborsClassifier`, which made the
"Custom KNN" entry in the model comparison identical to the "KNN" entry beside it --
two rows, one algorithm, and nothing custom about either.

The algorithm has no training step. `fit` stores the data; all the work happens at
predict time, where every test point is compared against every training point. That is
what makes kNN cheap to fit and expensive to predict, and the distance matrix is the
reason: `n_test x n_train` floats. At 8,000 test rows against 32,000 training rows that
single array would be 2GB, so predictions are computed in chunks and only the chunk is
ever held.

Tie-breaking matches scikit-learn so the two agree exactly, which is what the tests
assert: neighbours are ordered by distance with ties broken by lower training index
(a stable sort), and a tie in the class vote goes to the lower class index (`argmax`).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Rows of the distance matrix computed at once. 512 x n_train float64 is a few hundred
# MB at most, which the 512MB deploy target can hold; the whole matrix could not.
PREDICT_CHUNK_ROWS = 512


def _pairwise_distances(A: np.ndarray, B: np.ndarray, metric: str, p: float) -> np.ndarray:
    """Return the distance from every row of `A` to every row of `B`.

    Euclidean is expanded as ||a||^2 - 2a.b + ||b||^2 so the inner term is a single
    matrix multiply rather than a Python loop -- the same identity BLAS-backed libraries
    use. The others need the full difference tensor, which is why they are slower.
    """
    if metric == "euclidean":
        # Clipped because the expansion can produce small negative values from floating
        # point cancellation when two points coincide, and sqrt of those is nan.
        squared = (
            np.sum(A ** 2, axis=1)[:, None]
            - 2.0 * (A @ B.T)
            + np.sum(B ** 2, axis=1)[None, :]
        )
        return np.sqrt(np.maximum(squared, 0.0))

    diff = np.abs(A[:, None, :] - B[None, :, :])
    if metric == "manhattan":
        return diff.sum(axis=2)
    if metric == "chebyshev":
        return diff.max(axis=2)
    if metric == "minkowski":
        return (diff ** p).sum(axis=2) ** (1.0 / p)
    raise ValueError(f"Unsupported metric: {metric}")


class _BaseKNN:
    """Shared neighbour search. Subclasses decide what to do with the neighbours."""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", p: float = 2.0) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseKNN":
        """Memorise the training set. kNN has no parameters to estimate."""
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y)
        if len(self._X) < self.n_neighbors:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the {len(self._X)} training rows."
            )
        return self

    def _neighbour_indices(self, X: np.ndarray) -> np.ndarray:
        """Return the indices of the k nearest training rows for each row of `X`."""
        if self._X is None:
            raise RuntimeError("Call fit before predict.")

        query = np.asarray(X, dtype=float)
        out = np.empty((len(query), self.n_neighbors), dtype=np.intp)

        for start in range(0, len(query), PREDICT_CHUNK_ROWS):
            chunk = query[start:start + PREDICT_CHUNK_ROWS]
            distances = _pairwise_distances(chunk, self._X, self.metric, self.p)

            # argpartition finds the k smallest in O(n) without ordering the rest, then
            # only those k are sorted. A full argsort of every row would be the dominant
            # cost on a large training set.
            candidates = np.argpartition(distances, self.n_neighbors - 1, axis=1)
            candidates = candidates[:, :self.n_neighbors]
            picked = np.take_along_axis(distances, candidates, axis=1)
            # Stable, so equidistant neighbours keep training order -- this is what makes
            # the result identical to scikit-learn's rather than merely equivalent.
            order = np.argsort(picked, axis=1, kind="stable")
            out[start:start + len(chunk)] = np.take_along_axis(candidates, order, axis=1)

        return out


class CustomKNN(_BaseKNN):
    """k-nearest-neighbours classifier: the majority class among the k nearest rows."""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", p: float = 2.0) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, p=p)
        self.classes_: np.ndarray | None = None
        logger.info("CustomKNN initialised (k=%d, metric=%s).", n_neighbors, metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomKNN":
        """Store the training data and the set of classes."""
        super().fit(X, y)
        self.classes_ = np.unique(self._y)
        logger.info("CustomKNN stored %d training rows.", len(self._X))
        return self

    def _vote_counts(self, X: np.ndarray) -> np.ndarray:
        """Return, for each row, how many of its k neighbours belong to each class."""
        assert self.classes_ is not None and self._y is not None
        neighbours = self._y[self._neighbour_indices(X)]
        # Compare against the class list once rather than looping over rows.
        return (neighbours[:, :, None] == self.classes_[None, None, :]).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most common class among each row's k nearest neighbours."""
        assert self.classes_ is not None
        return self.classes_[np.argmax(self._vote_counts(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the share of each row's k neighbours that belong to each class.

        These are neighbour fractions, not calibrated probabilities: with k=5 the only
        values possible are 0, 0.2, 0.4, 0.6, 0.8 and 1.
        """
        return self._vote_counts(X) / float(self.n_neighbors)

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatibility, so this can be cloned and cross-validated."""
        return {"n_neighbors": self.n_neighbors, "metric": self.metric, "p": self.p}

    def set_params(self, **params) -> "CustomKNN":
        """scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CustomKNNRegressor(_BaseKNN):
    """k-nearest-neighbours regressor: the mean target of the k nearest rows."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomKNNRegressor":
        """Store the training data with a float target."""
        super().fit(X, np.asarray(y, dtype=float))
        logger.info("CustomKNNRegressor stored %d training rows.", len(self._X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the mean target value over each row's k nearest neighbours."""
        assert self._y is not None
        return self._y[self._neighbour_indices(X)].mean(axis=1)

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn compatibility."""
        return {"n_neighbors": self.n_neighbors, "metric": self.metric, "p": self.p}

    def set_params(self, **params) -> "CustomKNNRegressor":
        """scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
