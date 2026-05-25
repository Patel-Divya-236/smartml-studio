"""Unit tests for the PreprocessingAdvisor class."""

import pytest
from src.advisors.preprocessing_advisor import PreprocessingAdvisor
from src.profiling.dataset_profiler import DatasetProfiler


def test_preprocessing_advisor_recommendations(sample_classification_df):
    """Test that preprocessing advisor generates appropriate recommendations."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target")
    profile = profiler.compute_profile()

    advisor = PreprocessingAdvisor()
    recs = advisor.recommend(profile)

    assert len(recs) > 0
    # Encoding check (category)
    encoding_recs = [r for r in recs if r.category == "encoding" and r.metadata.get("column") == "category"]
    assert len(encoding_recs) == 1
    assert encoding_recs[0].metadata["action"] == "onehot"

    # Scaling check (income)
    scaling_recs = [r for r in recs if r.category == "scaling" and r.metadata.get("column") == "income"]
    assert len(scaling_recs) == 1
    assert scaling_recs[0].metadata["action"] == "standard"
