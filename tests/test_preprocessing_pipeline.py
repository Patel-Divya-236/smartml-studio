"""Unit tests for the PreprocessingPipeline.

The load-bearing tests here are the leakage checks: they assert that every fitted
statistic comes from the training rows alone.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


@pytest.fixture
def split_frame(sample_classification_df):
    """Return (full, train, test) frames split 70/30 without shuffling."""
    df = sample_classification_df.drop(columns=["target"])
    return df, df.iloc[:70], df.iloc[70:]


def test_scaler_is_fitted_on_train_rows_only(split_frame):
    """StandardScaler statistics must come from train, never from the full dataset."""
    full, train, test = split_frame

    pipeline = PreprocessingPipeline(
        target_column="target",
        impute_config={},
        encode_config={},
        scale_config={"income": "Standard"},
    )
    pipeline.fit(train)

    fitted_mean = pipeline._scalers["income"].mean_[0]

    # It matches the training mean...
    assert fitted_mean == pytest.approx(train["income"].mean())
    # ...and is genuinely different from the full-dataset mean, so a regression
    # back to fit-on-everything would fail this test rather than pass silently.
    assert fitted_mean != pytest.approx(full["income"].mean())


def test_transform_uses_train_statistics_on_test_rows(split_frame):
    """Test rows are standardised with the train mean, so their own mean is not 0."""
    _, train, test = split_frame

    pipeline = PreprocessingPipeline("target", {}, {}, {"income": "Standard"})
    pipeline.fit(train)

    train_out = pipeline.transform(train)
    test_out = pipeline.transform(test)

    # Training rows centre on zero by construction.
    assert train_out["income"].mean() == pytest.approx(0.0, abs=1e-9)
    # Test rows are shifted by the train statistics, so they do not centre exactly.
    assert test_out["income"].mean() != pytest.approx(0.0, abs=1e-9)


def test_train_and_test_share_identical_columns(split_frame):
    """One-hot encoding must not produce a different column set for the test split."""
    _, train, test = split_frame

    pipeline = PreprocessingPipeline(
        "target",
        impute_config={},
        encode_config={"category": "One-Hot"},
        scale_config={"age": "Standard"},
    )
    train_out = pipeline.fit_transform(train)
    test_out = pipeline.transform(test)

    assert list(train_out.columns) == list(test_out.columns)
    assert list(train_out.columns) == pipeline.feature_names_out


def test_unseen_test_category_does_not_add_a_column():
    """A category present only at transform time is ignored, not encoded."""
    train = pd.DataFrame({"colour": ["red", "blue", "red", "blue"], "n": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"colour": ["red", "green"], "n": [5.0, 6.0]})

    pipeline = PreprocessingPipeline("target", {}, {"colour": "One-Hot"}, {})
    train_out = pipeline.fit_transform(train)
    test_out = pipeline.transform(test)

    assert list(train_out.columns) == list(test_out.columns)
    assert len(test_out) == 2
    # 'green' was never seen, so every one-hot column is 0 for that row.
    encoded_cols = [c for c in test_out.columns if c.startswith("colour")]
    assert test_out.loc[1, encoded_cols].sum() == 0.0


def test_imputation_fill_value_comes_from_train():
    """The median used to fill test gaps is the train median."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [np.nan, 100.0]})

    pipeline = PreprocessingPipeline("target", {"a": "Median"}, {}, {})
    pipeline.fit(train)
    test_out = pipeline.transform(test)

    assert test_out["a"].iloc[0] == pytest.approx(3.0)  # train median, not test median


def test_drop_column_action_removes_column(split_frame):
    """Columns marked 'Drop Column' are absent from both outputs."""
    _, train, test = split_frame

    pipeline = PreprocessingPipeline("target", {"score": "Drop Column"}, {}, {})
    train_out = pipeline.fit_transform(train)
    test_out = pipeline.transform(test)

    assert "score" not in train_out.columns
    assert "score" not in test_out.columns
    assert pipeline.dropped_columns == ["score"]


def test_row_counts_are_preserved(split_frame):
    """Transformation never adds or removes rows."""
    _, train, test = split_frame

    pipeline = PreprocessingPipeline("target", {}, {"category": "Ordinal/Label"}, {"age": "Robust"})
    assert len(pipeline.fit_transform(train)) == len(train)
    assert len(pipeline.transform(test)) == len(test)


def test_log1p_shift_is_learned_at_fit_time():
    """The negative-value shift must be reused at transform, not recomputed per frame."""
    train = pd.DataFrame({"v": [-3.0, 0.0, 5.0, 10.0]})
    test = pd.DataFrame({"v": [-2.0, 4.0]})

    pipeline = PreprocessingPipeline("target", {}, {}, {"v": "Log1p"})
    pipeline.fit(train)

    # min is -3, so the shift is 4.0 and is stored rather than derived per call.
    assert pipeline._log1p_shifts["v"] == pytest.approx(4.0)

    test_out = pipeline.transform(test)
    assert np.isfinite(test_out["v"]).all()


def test_transform_before_fit_raises():
    """Using an unfitted pipeline is an error, not a silent no-op."""
    pipeline = PreprocessingPipeline("target", {}, {}, {})
    with pytest.raises(ValueError, match="must be fitted"):
        pipeline.transform(pd.DataFrame({"a": [1.0]}))
