"""Unit tests for the CustomSVM from scratch implementation.

Compares predictions and fit behavior with sklearn.svm.SVC.
"""

import numpy as np
import pytest
from sklearn.svm import SVC

from src.models.custom_svm import CustomSVM


def test_custom_svm_linear_kernel():
    """Verify linear CustomSVM works on simple binary dataset and aligns with SVC."""
    # Toy dataset
    X = np.array([
        [2.0, 3.0],
        [1.0, 1.0],
        [2.3, 2.7],
        [0.8, 1.2],
        [3.0, 3.5],
        [0.5, 0.5]
    ])
    y = np.array([1, 0, 1, 0, 1, 0])

    # Fit custom SVM
    svm_custom = CustomSVM(kernel="linear", C=1.0, learning_rate=0.1, n_iters=100)
    svm_custom.fit(X, y)
    preds_custom = svm_custom.predict(X)

    # Fit sklearn SVM
    svm_sklearn = SVC(kernel="linear", C=1.0)
    svm_sklearn.fit(X, y)
    preds_sklearn = svm_sklearn.predict(X)

    # Assert correctness
    # Check that accuracy on toy set is reasonable (>= 0.8)
    acc = np.mean(preds_custom == y)
    assert acc >= 0.83
    # Check prediction matching with sklearn
    matching_ratio = np.mean(preds_custom == preds_sklearn)
    assert matching_ratio >= 0.83


def test_custom_svm_rbf_kernel():
    """Verify RBF kernel CustomSVM works and trains correctly."""
    X = np.array([
        [2.0, 3.0],
        [1.0, 1.0],
        [2.3, 2.7],
        [0.8, 1.2],
        [3.0, 3.5],
        [0.5, 0.5]
    ])
    y = np.array([1, 0, 1, 0, 1, 0])

    svm_custom = CustomSVM(kernel="rbf", C=1.0, gamma=0.5, learning_rate=0.01, n_iters=100)
    svm_custom.fit(X, y)
    preds_custom = svm_custom.predict(X)

    assert len(preds_custom) == 6
    assert np.mean(preds_custom == y) >= 0.83
