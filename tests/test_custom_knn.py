"""Unit tests for the CustomKNN from scratch implementation.

Compares predictions and fit behavior with sklearn.neighbors.KNeighborsClassifier.
"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from src.models.custom_knn import CustomKNN


def test_custom_knn_euclidean(sample_classification_arrays):
    """Verify Euclidean CustomKNN matches KNeighborsClassifier predictions exactly."""
    X, y = sample_classification_arrays

    # Fit custom
    knn_custom = CustomKNN(n_neighbors=3, metric="euclidean")
    knn_custom.fit(X, y)
    preds_custom = knn_custom.predict(X)

    # Fit sklearn
    knn_sklearn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_sklearn.fit(X, y)
    preds_sklearn = knn_sklearn.predict(X)

    # They should match 100% since calculations are mathematically identical
    assert np.array_equal(preds_custom, preds_sklearn)


def test_custom_knn_manhattan(sample_classification_arrays):
    """Verify Manhattan CustomKNN matches KNeighborsClassifier predictions exactly."""
    X, y = sample_classification_arrays

    # Fit custom
    knn_custom = CustomKNN(n_neighbors=5, metric="manhattan")
    knn_custom.fit(X, y)
    preds_custom = knn_custom.predict(X)

    # Fit sklearn
    knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
    knn_sklearn.fit(X, y)
    preds_sklearn = knn_sklearn.predict(X)

    # They should match 100% since calculations are mathematically identical
    assert np.array_equal(preds_custom, preds_sklearn)
