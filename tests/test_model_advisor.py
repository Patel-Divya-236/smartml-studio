"""Unit tests for the ModelAdvisor class."""

import pytest
from src.advisors.model_advisor import ModelAdvisor
from src.profiling.dataset_profiler import DatasetProfiler


def test_model_advisor_recommendations(sample_classification_df):
    """Test that model advisor generates appropriate recommendations."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target", problem_type="Classification")
    profile = profiler.compute_profile()

    advisor = ModelAdvisor()
    recs = advisor.recommend(profile)

    assert len(recs) > 0
    # Classifier check
    model_names = [r.metadata.get("model_name") for r in recs]
    assert "XGBoost" in model_names
    assert "Random Forest" in model_names
    assert "SVM" in model_names
    assert "KNN" in model_names


def test_model_advisor_recommends_regressors_for_continuous_target(sample_regression_df):
    """A regression target must reach the regression branch.

    Regression previously never fired: the branch keyed off a non-empty class_balance,
    which value_counts() populated for any target at all.
    """
    df = sample_regression_df
    profiler = DatasetProfiler(df, target_column="target", problem_type="Regression")
    profile = profiler.compute_profile()

    recs = ModelAdvisor().recommend(profile)
    model_names = [r.metadata.get("model_name") for r in recs]

    assert "Linear Regression" in model_names
    # Naive Bayes is classification-only and raises inside ModelTrainer.
    assert "Naive Bayes" not in model_names
    assert all(r.category == "Regression" for r in recs)


def test_model_advisor_treats_time_series_as_continuous(sample_regression_df):
    """Time Series shares the continuous-target branch; no classifiers are offered."""
    profiler = DatasetProfiler(sample_regression_df, target_column="target", problem_type="Time Series")
    recs = ModelAdvisor().recommend(profiler.compute_profile())

    assert "Naive Bayes" not in [r.metadata.get("model_name") for r in recs]
    assert all(r.category == "Regression" for r in recs)
