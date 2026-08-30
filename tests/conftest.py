"""Shared pytest fixtures for SmartML Studio tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """Small synthetic classification dataset for unit tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(18, 80, size=n),
        "income": np.random.normal(50000, 15000, size=n).round(2),
        "score": np.random.uniform(0, 100, size=n).round(2),
        "category": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    })


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """Small synthetic regression dataset for unit tests."""
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, size=n)
    X2 = np.random.normal(0, 1, size=n)
    noise = np.random.normal(0, 0.5, size=n)
    return pd.DataFrame({
        "feature_1": X1.round(4),
        "feature_2": X2.round(4),
        "feature_3": np.random.uniform(-5, 5, size=n).round(4),
        "target": (3 * X1 + 2 * X2 + noise).round(4),
    })


@pytest.fixture
def sample_classification_arrays(
    sample_classification_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays from the classification fixture."""
    df = sample_classification_df
    X = df.drop(columns=["target", "category"]).values
    y = df["target"].values
    return X, y
