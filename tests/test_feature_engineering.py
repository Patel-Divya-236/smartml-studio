"""Unit tests for the FeatureEngineeringPipeline.

SelectKBest is the sharpest leak in this module — fitting it on the full dataset
picks features using the test set's target values — so it gets the most attention.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineeringPipeline


@pytest.fixture
def split_regression(sample_regression_df):
    """Return (full_X, train_X, test_X, train_y) split 70/30 without shuffling."""
    df = sample_regression_df
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, X.iloc[:70], X.iloc[70:], y.iloc[:70]


def test_pca_is_fitted_on_train_rows_only(split_regression):
    """PCA's learned centre must be the train mean, not the full-dataset mean."""
    full_X, train_X, _, train_y = split_regression

    pipeline = FeatureEngineeringPipeline(
        {"pca_active": True, "pca_components": 2}, problem_type="Regression"
    )
    pipeline.fit(train_X, train_y)

    np.testing.assert_allclose(pipeline._pca.mean_, train_X.mean().to_numpy(), rtol=1e-9)
    # And is distinguishable from the full-dataset centre.
    assert not np.allclose(pipeline._pca.mean_, full_X.mean().to_numpy())


def test_pca_output_shape_and_columns(split_regression):
    """Train and test are projected onto the same components."""
    _, train_X, test_X, train_y = split_regression

    pipeline = FeatureEngineeringPipeline(
        {"pca_active": True, "pca_components": 2}, problem_type="Regression"
    )
    train_out = pipeline.fit_transform(train_X, train_y)
    test_out = pipeline.transform(test_X)

    assert list(train_out.columns) == ["PC1", "PC2"]
    assert list(test_out.columns) == ["PC1", "PC2"]
    assert len(test_out) == len(test_X)


def test_select_k_best_uses_train_target_only(split_regression):
    """Selected columns are decided at fit time and replayed verbatim on test."""
    _, train_X, test_X, train_y = split_regression

    pipeline = FeatureEngineeringPipeline(
        {"select_k_best_active": True, "select_k_best_k": 2, "select_k_best_method": "ANOVA"},
        problem_type="Regression",
    )
    train_out = pipeline.fit_transform(train_X, train_y)
    test_out = pipeline.transform(test_X)

    assert train_out.shape[1] == 2
    assert list(train_out.columns) == list(test_out.columns)
    # The choice is frozen on the pipeline, so transform cannot re-decide using test labels.
    assert pipeline._selected_columns == list(train_out.columns)


def test_select_k_best_picks_the_informative_features():
    """The fixture's target is 3*f1 + 2*f2 + noise, so f3 should be dropped."""
    rng = np.random.default_rng(0)
    n = 200
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)  # pure noise, unrelated to y
    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    y = pd.Series(3 * f1 + 2 * f2 + rng.normal(0, 0.1, size=n))

    pipeline = FeatureEngineeringPipeline(
        {"select_k_best_active": True, "select_k_best_k": 2}, problem_type="Regression"
    )
    out = pipeline.fit_transform(X, y)

    assert set(out.columns) == {"f1", "f2"}


def test_low_variance_filter_drops_constant_columns():
    """A constant column carries no signal and is removed from both splits."""
    train = pd.DataFrame({"varies": [1.0, 5.0, 9.0, 2.0], "constant": [7.0, 7.0, 7.0, 7.0]})
    test = pd.DataFrame({"varies": [3.0, 8.0], "constant": [7.0, 7.0]})
    y = pd.Series([0, 1, 0, 1])

    pipeline = FeatureEngineeringPipeline(
        {"low_variance_active": True, "low_variance_threshold": 0.01},
        problem_type="Classification",
    )
    train_out = pipeline.fit_transform(train, y)
    test_out = pipeline.transform(test)

    assert "constant" not in train_out.columns
    assert list(train_out.columns) == list(test_out.columns)


def test_polynomial_expansion_matches_across_splits(split_regression):
    """Train and test receive the same expanded feature set."""
    _, train_X, test_X, train_y = split_regression

    pipeline = FeatureEngineeringPipeline(
        {"poly_active": True, "poly_degree": 2, "poly_interaction_only": False},
        problem_type="Regression",
    )
    train_out = pipeline.fit_transform(train_X, train_y)
    test_out = pipeline.transform(test_X)

    assert train_out.shape[1] > train_X.shape[1]
    assert list(train_out.columns) == list(test_out.columns)


def test_no_active_steps_is_a_passthrough(split_regression):
    """With every step disabled the frame comes back unchanged."""
    _, train_X, test_X, train_y = split_regression

    pipeline = FeatureEngineeringPipeline({}, problem_type="Regression")
    train_out = pipeline.fit_transform(train_X, train_y)

    assert list(train_out.columns) == list(train_X.columns)
    pd.testing.assert_frame_equal(train_out, train_X)


def test_transform_before_fit_raises():
    """Using an unfitted pipeline is an error."""
    pipeline = FeatureEngineeringPipeline({"pca_active": True}, problem_type="Regression")
    with pytest.raises(ValueError, match="must be fitted"):
        pipeline.transform(pd.DataFrame({"a": [1.0]}))
