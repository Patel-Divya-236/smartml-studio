"""Unit tests for the ModelAdvisor class."""

import pytest
from src.advisors.model_advisor import ModelAdvisor
from src.profiling.dataset_profiler import DatasetProfiler


def test_model_advisor_recommendations(sample_classification_df):
    """Test that model advisor generates appropriate recommendations."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target")
    profile = profiler.compute_profile()

    advisor = ModelAdvisor()
    recs = advisor.recommend(profile)

    assert len(recs) > 0
    # Classifier check
    model_names = [r.metadata.get("model_name") for r in recs]
    assert "XGBoost" in model_names
    assert "Random Forest" in model_names
    assert "Custom SVM" in model_names
    assert "Custom KNN" in model_names
