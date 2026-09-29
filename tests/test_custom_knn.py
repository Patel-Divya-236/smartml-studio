"""Tests for the from-scratch kNN in src/models/custom_knn.py.

kNN is deterministic, so "close to sklearn" is not the bar -- an independent correct
implementation must agree with `KNeighborsClassifier` exactly, ties included. These
tests used to compare a wrapper around `KNeighborsClassifier` against
`KNeighborsClassifier`, which is a comparison that cannot fail.
"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from src.models.custom_knn import PREDICT_CHUNK_ROWS, CustomKNN, CustomKNNRegressor


def test_it_is_not_a_wrapper_around_sklearn(sample_classification_arrays):
    """The point of the class. Guards against it quietly becoming a wrapper again."""
    X, y = sample_classification_arrays
    model = CustomKNN(n_neighbors=3).fit(X, y)

    for attribute in vars(model).values():
        assert not isinstance(attribute, KNeighborsClassifier)


@pytest.mark.parametrize("metric,k", [("euclidean", 3), ("manhattan", 5), ("chebyshev", 4)])
def test_predictions_match_sklearn_exactly(sample_classification_arrays, metric, k):
    """Same distances, same neighbours, same vote -- so the labels must be identical."""
    X, y = sample_classification_arrays

    mine = CustomKNN(n_neighbors=k, metric=metric).fit(X, y).predict(X)
    theirs = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X, y).predict(X)

    assert np.array_equal(mine, theirs)


def test_minkowski_matches_sklearn_at_p_equals_three(sample_classification_arrays):
    X, y = sample_classification_arrays

    mine = CustomKNN(n_neighbors=5, metric="minkowski", p=3.0).fit(X, y).predict(X)
    theirs = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=3).fit(X, y).predict(X)

    assert np.array_equal(mine, theirs)


def test_chunking_does_not_change_the_answer():
    """Predictions are computed in blocks so the distance matrix stays bounded.

    At 8,000 test rows against 32,000 training rows the full matrix would be 2GB on a
    512MB instance. Asking for more rows than one chunk holds must give the same answer
    as asking for fewer.
    """
    rng = np.random.default_rng(0)
    rows = PREDICT_CHUNK_ROWS * 2 + 37
    X_train = rng.normal(size=(400, 6))
    y_train = rng.integers(0, 3, 400)
    X_test = rng.normal(size=(rows, 6))

    model = CustomKNN(n_neighbors=5).fit(X_train, y_train)
    whole = model.predict(X_test)
    piecewise = np.concatenate([model.predict(X_test[i:i + 100]) for i in range(0, rows, 100)])

    assert np.array_equal(whole, piecewise)
    assert np.array_equal(whole, KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).predict(X_test))


def test_probabilities_are_neighbour_fractions(sample_classification_arrays):
    """With k=4 the only values possible are 0, 0.25, 0.5, 0.75 and 1."""
    X, y = sample_classification_arrays

    proba = CustomKNN(n_neighbors=4).fit(X, y).predict_proba(X)

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(proba)) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert np.allclose(proba, KNeighborsClassifier(n_neighbors=4).fit(X, y).predict_proba(X))


def test_asking_for_more_neighbours_than_rows_is_refused():
    """Silently shrinking k would change the model the user asked for."""
    with pytest.raises(ValueError, match="exceeds"):
        CustomKNN(n_neighbors=10).fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))


def test_regressor_averages_the_neighbours(sample_regression_df):
    """The regression path used to fall back to sklearn's KNeighborsRegressor."""
    X = sample_regression_df.drop(columns=["target"]).values
    y = sample_regression_df["target"].values

    mine = CustomKNNRegressor(n_neighbors=5).fit(X, y).predict(X)
    theirs = KNeighborsRegressor(n_neighbors=5).fit(X, y).predict(X)

    assert np.allclose(mine, theirs)
