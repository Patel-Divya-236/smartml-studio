"""Unit tests for the DatasetProfiler class."""

import pandas as pd
import numpy as np
import pytest

from src.profiling.dataset_profiler import DatasetProfiler


def test_dataset_profiler_classification(sample_classification_df):
    """Test profiler logic on a synthetic classification dataset."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target")
    profile = profiler.compute_profile()

    # Shape checks
    assert profile["shape"] == (100, 5)
    assert profile["duplicates"] == 0

    # Missing counts
    assert all(count == 0 for count in profile["missing_values"].values())

    # Dtypes
    assert "int" in profile["dtypes"]["age"]
    assert profile["dtypes"]["category"] in ["object", "str"]

    # Cardinality
    assert profile["cardinality"]["category"] == 3

    # Outliers
    assert "income" in profile["outliers"]
    assert "count" in profile["outliers"]["income"]

    # Class balance
    assert 0 in profile["class_balance"]
    assert 1 in profile["class_balance"]


def test_dataset_profiler_missing_values():
    """Test profiler handles missing values correctly."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": ["x", "y", "z", "w", None],
        "target": [0, 1, 0, 1, 0]
    })
    profiler = DatasetProfiler(df, target_column="target")
    profile = profiler.compute_profile()

    assert profile["missing_values"]["a"] == 1
    assert profile["missing_values"]["b"] == 1
    assert profile["missing_pct"]["a"] == 20.0
    assert profile["missing_pct"]["b"] == 20.0
