"""Unit tests for the DatasetProfiler class."""

import pandas as pd
import numpy as np
import pytest

from src.profiling.dataset_profiler import DatasetProfiler, detect_problem_type


def test_dataset_profiler_classification(sample_classification_df):
    """Test profiler logic on a synthetic classification dataset."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target", problem_type="Classification")
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


def test_profile_carries_problem_type(sample_classification_df):
    """The profile exposes the task type so advisors can branch on it."""
    profiler = DatasetProfiler(sample_classification_df, target_column="target", problem_type="Classification")
    assert profiler.compute_profile()["problem_type"] == "Classification"


def test_problem_type_is_inferred_when_not_supplied(sample_regression_df):
    """Omitting problem_type falls back to detect_problem_type rather than None."""
    profiler = DatasetProfiler(sample_regression_df, target_column="target")
    assert profiler.compute_profile()["problem_type"] == "Regression"


def test_explicit_problem_type_overrides_inference(sample_regression_df):
    """The user's confirmed choice wins over auto-detection."""
    profiler = DatasetProfiler(sample_regression_df, target_column="target", problem_type="Time Series")
    assert profiler.compute_profile()["problem_type"] == "Time Series"


def test_class_balance_is_empty_for_regression(sample_regression_df):
    """A continuous target must not produce a class-distribution map.

    value_counts() on a continuous column returns one entry per distinct value, which
    previously made every advisor treat regression datasets as classification.
    """
    profiler = DatasetProfiler(sample_regression_df, target_column="target", problem_type="Regression")
    profile = profiler.compute_profile()

    assert profile["class_balance"] == {}
    # Summary statistics are still available for the regression UI.
    assert set(profile["target_summary"]) == {"mean", "median", "min", "max"}


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, 0, 1, 1], "Classification"),          # few unique numeric values
        (list(range(50)), "Regression"),               # many unique numeric values
        (["a", "b", "a", "c", "b"], "Classification"), # categorical
    ],
)
def test_detect_problem_type(values, expected):
    """Task-type detection covers the numeric, categorical, and continuous cases."""
    df = pd.DataFrame({"target": values})
    assert detect_problem_type(df, "target")[0] == expected
