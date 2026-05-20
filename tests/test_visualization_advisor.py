"""Unit tests for the VisualizationAdvisor class."""

import pytest
from src.advisors.visualization_advisor import VisualizationAdvisor
from src.profiling.dataset_profiler import DatasetProfiler


def test_visualization_advisor_recommendations(sample_classification_df):
    """Test that visualization advisor generates proper recommendations."""
    df = sample_classification_df
    profiler = DatasetProfiler(df, target_column="target")
    profile = profiler.compute_profile()

    advisor = VisualizationAdvisor()
    recs = advisor.recommend(profile)

    assert len(recs) > 0
    # Heatmap check
    heatmap_recs = [r for r in recs if r.metadata.get("chart_type") == "correlation_heatmap"]
    assert len(heatmap_recs) == 1
    assert heatmap_recs[0].confidence_score == 0.95

    # Pie chart check (target_pie_chart)
    pie_recs = [r for r in recs if r.metadata.get("chart_type") == "target_pie_chart"]
    assert len(pie_recs) == 1
    assert pie_recs[0].confidence_score == 0.90
