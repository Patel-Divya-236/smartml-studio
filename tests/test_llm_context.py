"""Unit tests for the LLM payload builders.

The load-bearing test here is the privacy one: it asserts that the uploaded dataset
never reaches a prompt. Because every prompt is assembled in src/llm/context.py, that
property is enforceable in a single file and checkable in a single test.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import SETTINGS
from src.llm.context import (
    build_prediction_payload,
    build_report_payload,
    summarise_profile,
)
from src.profiling.dataset_profiler import DatasetProfiler

# Distinctive values that would be unmistakable if they leaked into a prompt.
SENTINELS = [987654.321, 876543.219, 765432.198, 654321.987, 543219.876]


@pytest.fixture
def sentinel_df() -> pd.DataFrame:
    """A dataset whose feature values are individually traceable."""
    return pd.DataFrame(
        {
            "salary": SENTINELS,
            "age": [31, 42, 53, 64, 75],
            "region": ["north", "south", "north", "east", "south"],
            "target": [0, 1, 0, 1, 0],
        }
    )


# ── Privacy ───────────────────────────────────────────────────────────


def test_profile_summary_leaks_no_row_values(sentinel_df):
    """Aggregates only — no individual cell from the dataset reaches the prompt."""
    profile = DatasetProfiler(sentinel_df, "target", problem_type="Classification").compute_profile()
    summary = summarise_profile(profile)

    for sentinel in SENTINELS:
        assert str(int(sentinel)) not in summary, f"{sentinel} leaked into the profile summary"


def test_report_payload_leaks_no_row_values(sentinel_df):
    """The report prompt carries aggregates and metrics, never dataset rows."""
    profile = DatasetProfiler(sentinel_df, "target", problem_type="Classification").compute_profile()
    comparison = pd.DataFrame(
        [{"Model Name": "XGBoost", "Accuracy": 0.91, "F1-Score": 0.90}]
    )

    payload = build_report_payload(
        dataset_name="sentinels.csv",
        target_column="target",
        problem_type="Classification",
        profile=profile,
        comparison=comparison,
    )

    for sentinel in SENTINELS:
        assert str(int(sentinel)) not in payload, f"{sentinel} leaked into the report payload"
    assert "XGBoost" in payload  # the metrics table is still present


def test_prediction_payload_contains_only_the_explained_row():
    """Explaining one sample must not disclose any other sample's values."""
    feature_names = ["salary", "age", "score"]
    explained_row = np.array([1.5, -0.5, 2.0])
    other_rows_sentinel = 424242.0

    payload = build_prediction_payload(
        feature_names=feature_names,
        feature_values=explained_row,
        shap_values=np.array([0.4, -0.2, 0.1]),
        predicted_label=1,
    )

    assert str(int(other_rows_sentinel)) not in payload
    assert "salary" in payload


# ── Payload construction ──────────────────────────────────────────────


def test_prediction_payload_truncates_to_configured_top_n():
    """Only the top-N attributions by magnitude are sent, bounding prompt size."""
    n_features = SETTINGS.LLM_MAX_SHAP_FEATURES + 8
    names = [f"feature_{i}" for i in range(n_features)]

    payload = build_prediction_payload(
        feature_names=names,
        feature_values=np.arange(n_features, dtype=float),
        shap_values=np.arange(n_features, dtype=float),
        predicted_label=1,
    )

    # Count table rows rather than substring-matching names, since "feature_1"
    # is a substring of "feature_10".
    listed = [ln for ln in payload.splitlines() if ln.startswith("feature_")]
    assert len(listed) == SETTINGS.LLM_MAX_SHAP_FEATURES
    assert "further features" in payload


def test_prediction_payload_orders_by_absolute_contribution():
    """The largest absolute contribution is listed first, regardless of sign."""
    payload = build_prediction_payload(
        feature_names=["small", "large_negative", "medium"],
        feature_values=np.array([1.0, 2.0, 3.0]),
        shap_values=np.array([0.01, -0.90, 0.30]),
        predicted_label=0,
    )
    body = payload.split("contribution")[-1]
    assert body.index("large_negative") < body.index("medium") < body.index("small")


def test_prediction_payload_handles_empty_input():
    """No usable attributions degrades to a message rather than raising."""
    payload = build_prediction_payload(
        feature_names=[],
        feature_values=np.array([]),
        shap_values=np.array([]),
        predicted_label=1,
    )
    assert "No feature attributions" in payload


def test_report_payload_without_models_is_still_valid(sentinel_df):
    """A report requested before training still produces a usable prompt."""
    profile = DatasetProfiler(sentinel_df, "target", problem_type="Classification").compute_profile()
    payload = build_report_payload(
        dataset_name="d.csv",
        target_column="target",
        problem_type="Classification",
        profile=profile,
        comparison=None,
    )
    assert "No models were evaluated." in payload


def test_report_payload_includes_ensemble_details_when_present(sentinel_df):
    """Ensemble configuration reaches the summary when one was used."""
    profile = DatasetProfiler(sentinel_df, "target", problem_type="Classification").compute_profile()
    payload = build_report_payload(
        dataset_name="d.csv",
        target_column="target",
        problem_type="Classification",
        profile=profile,
        comparison=pd.DataFrame([{"Model Name": "RF", "Accuracy": 0.8}]),
        report_state={
            "prediction_mode": "Custom Hybrid Ensemble",
            "selected_model": "Custom Hybrid Ensemble",
            "ensemble_voting": "weighted",
            "ensemble_models": ["RF", "XGBoost"],
        },
    )
    assert "weighted" in payload
    assert "RF, XGBoost" in payload


def test_profile_summary_reports_regression_target(sample_regression_df):
    """A continuous target is summarised by distribution, not class balance."""
    profile = DatasetProfiler(sample_regression_df, "target", problem_type="Regression").compute_profile()
    summary = summarise_profile(profile)
    assert "Target distribution" in summary
    assert "class balance" not in summary.lower()


def test_prediction_payload_accepts_multiclass_base_value():
    """A per-class baseline array must not raise — multiclass explainers return one."""
    payload = build_prediction_payload(
        feature_names=["a", "b"],
        feature_values=np.array([1.0, 2.0]),
        shap_values=np.array([0.3, -0.1]),
        predicted_label=2,
        base_value=np.array([0.11, 0.22, 0.33]),
    )
    assert "Model baseline" in payload
    assert "0.1100" in payload


def test_prediction_payload_tolerates_unusable_base_value():
    """A baseline that cannot be coerced is omitted rather than raising."""
    payload = build_prediction_payload(
        feature_names=["a"],
        feature_values=np.array([1.0]),
        shap_values=np.array([0.3]),
        predicted_label=1,
        base_value="not-a-number",
    )
    assert "Model baseline" not in payload
