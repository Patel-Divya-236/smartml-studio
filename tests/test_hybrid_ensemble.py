"""Unit tests for HybridEnsemble."""

import numpy as np
import pytest
from src.ensemble.hybrid_ensemble import HybridEnsemble


class MockModel:
    """Mock model that returns predefined predictions."""
    def __init__(self, preds: np.ndarray) -> None:
        self.preds = preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.preds


def test_ensemble_classification_majority() -> None:
    # 3 models, 5 samples
    m1 = MockModel(np.array([0, 1, 0, 1, 0]))
    m2 = MockModel(np.array([0, 0, 0, 1, 1]))
    m3 = MockModel(np.array([1, 1, 0, 0, 1]))
    
    ensemble = HybridEnsemble(voting="majority", problem_type="Classification")
    ensemble.fit([m1, m2, m3])
    
    X = np.zeros((5, 2))
    preds = ensemble.predict(X)
    
    # Mode predictions:
    # idx 0: [0, 0, 1] -> 0
    # idx 1: [1, 0, 1] -> 1
    # idx 2: [0, 0, 0] -> 0
    # idx 3: [1, 1, 0] -> 1
    # idx 4: [0, 1, 1] -> 1
    np.testing.assert_array_equal(preds, np.array([0, 1, 0, 1, 1]))


def test_ensemble_classification_weighted() -> None:
    m1 = MockModel(np.array([0, 1, 0]))
    m2 = MockModel(np.array([1, 0, 0]))
    
    # If weights are [2.0, 1.0], model 1 has more power
    ensemble = HybridEnsemble(voting="weighted", weights=[2.0, 1.0], problem_type="Classification")
    ensemble.fit([m1, m2])
    
    X = np.zeros((3, 2))
    preds = ensemble.predict(X)
    
    # Weighted predictions:
    # idx 0: model 1 predicts 0 (w=2.0), model 2 predicts 1 (w=1.0) -> 0 wins
    # idx 1: model 1 predicts 1 (w=2.0), model 2 predicts 0 (w=1.0) -> 1 wins
    # idx 2: both predict 0 -> 0 wins
    np.testing.assert_array_equal(preds, np.array([0, 1, 0]))


def test_ensemble_regression() -> None:
    m1 = MockModel(np.array([1.0, 2.0, 3.0]))
    m2 = MockModel(np.array([2.0, 4.0, 6.0]))
    
    ensemble = HybridEnsemble(voting="average", problem_type="Regression")
    ensemble.fit([m1, m2])
    
    X = np.zeros((3, 2))
    preds = ensemble.predict(X)
    np.testing.assert_array_almost_equal(preds, np.array([1.5, 3.0, 4.5]))

    # Weighted regression
    ensemble_weighted = HybridEnsemble(voting="weighted", weights=[3.0, 1.0], problem_type="Regression")
    ensemble_weighted.fit([m1, m2])
    preds_weighted = ensemble_weighted.predict(X)
    # (3*1.0 + 1*2.0)/4 = 1.25
    # (3*2.0 + 1*4.0)/4 = 2.5
    # (3*3.0 + 1*6.0)/4 = 3.75
    np.testing.assert_array_almost_equal(preds_weighted, np.array([1.25, 2.5, 3.75]))
