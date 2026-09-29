"""Tests for the from-scratch SVM in src/models/custom_svm.py.

These used to compare `CustomSVM` against `sklearn.svm.SVC` while `CustomSVM` *was* an
`SVC` underneath, so they compared sklearn to itself and could not fail. The comparisons
below are against a genuinely separate implementation, so the tolerances are real.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.custom_svm import CustomSVM, CustomSVR

# Well separated, but not separable by a boundary through the origin -- so a solver that
# never learns a bias fails this.
TOY_X = np.array([
    [2.0, 3.0], [1.0, 1.0], [2.3, 2.7],
    [0.8, 1.2], [3.0, 3.5], [0.5, 0.5],
])
TOY_Y = np.array([1, 0, 1, 0, 1, 0])


def _classification(n_classes: int, n_samples: int = 2000):
    X, y = make_classification(
        n_samples=n_samples, n_features=12, n_informative=8,
        n_classes=n_classes, n_clusters_per_class=1, random_state=0,
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=0)


def test_it_is_not_a_wrapper_around_sklearn():
    """The point of the class. Guards against it quietly becoming a wrapper again."""
    model = CustomSVM(kernel="linear", n_iters=200).fit(TOY_X, TOY_Y)

    for attribute in vars(model).values():
        assert not isinstance(attribute, SVC)
    for machine in model._machines:
        assert not any(isinstance(value, SVC) for value in vars(machine).values())


def test_linear_kernel_separates_the_toy_set():
    """Sub-gradient descent on the hinge loss should find this boundary exactly."""
    model = CustomSVM(kernel="linear", C=1.0, learning_rate=0.5, n_iters=300)
    model.fit(TOY_X, TOY_Y)

    assert np.array_equal(model.predict(TOY_X), TOY_Y)


def test_the_optimiser_actually_runs_the_iterations_it_is_given():
    """`learning_rate` and `n_iters` were accepted and ignored by the old wrapper.

    Too few steps must underfit. If a longer run does not beat a shorter one, nothing is
    being optimised and the parameters are decoration again.
    """
    starved = CustomSVM(kernel="linear", C=1.0, learning_rate=0.1, n_iters=20).fit(TOY_X, TOY_Y)
    converged = CustomSVM(kernel="linear", C=1.0, learning_rate=0.1, n_iters=2000).fit(TOY_X, TOY_Y)

    assert np.mean(starved.predict(TOY_X) == TOY_Y) < 1.0
    assert np.array_equal(converged.predict(TOY_X), TOY_Y)


@pytest.mark.parametrize("n_classes", [2, 4])
def test_rbf_kernel_comes_within_a_few_points_of_sklearn(n_classes):
    """The dual solver should land near libsvm's answer, not merely run."""
    X_train, X_test, y_train, y_test = _classification(n_classes)

    mine = CustomSVM(kernel="rbf", C=1.0, gamma=0.1, n_iters=500).fit(X_train, y_train)
    theirs = SVC(kernel="rbf", C=1.0, gamma=0.1).fit(X_train, y_train)

    mine_score = np.mean(mine.predict(X_test) == y_test)
    theirs_score = np.mean(theirs.predict(X_test) == y_test)
    assert mine_score >= theirs_score - 0.05, f"{mine_score:.3f} vs sklearn {theirs_score:.3f}"


def test_binary_rbf_does_not_collapse_to_one_class():
    """Regression test for the dual solver's original bug.

    A hand-picked step size overshot: every alpha hit a box boundary within five
    iterations, no support vector was left strictly inside the box to estimate the bias
    from, and the classifier put 99% of rows in one class for 49% accuracy -- chance.
    The step is now 1/lambda_max, and the bias is folded into the kernel rather than
    estimated. Accuracy alone would not catch a regression here, so the class balance of
    the predictions is what is asserted.
    """
    X_train, X_test, y_train, y_test = _classification(2)

    predicted = CustomSVM(kernel="rbf", C=1.0, gamma=0.1, n_iters=500).fit(X_train, y_train).predict(X_test)

    positive_rate = np.mean(predicted == 1)
    assert 0.3 < positive_rate < 0.7, f"{positive_rate:.0%} of rows predicted positive"
    assert np.mean(predicted == y_test) > 0.85


@pytest.mark.parametrize("n_classes", [2, 4])
def test_probabilities_are_shaped_and_normalised(n_classes):
    """Downstream metrics index these by class and assume the rows sum to one."""
    X_train, X_test, y_train, _ = _classification(n_classes, n_samples=600)

    model = CustomSVM(kernel="rbf", gamma=0.1, n_iters=200).fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    assert proba.shape == (len(X_test), n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_probabilities_rank_the_same_way_as_the_margin():
    """They are a squashed margin, so the ordering must survive the squashing."""
    X_train, X_test, y_train, _ = _classification(2, n_samples=600)

    model = CustomSVM(kernel="rbf", gamma=0.1, n_iters=200).fit(X_train, y_train)
    margins = model.decision_function(X_test)
    positive = model.predict_proba(X_test)[:, 1]

    assert np.array_equal(np.argsort(margins), np.argsort(positive))


def test_a_single_class_is_refused_rather_than_fitted():
    with pytest.raises(ValueError, match="at least two classes"):
        CustomSVM().fit(TOY_X, np.ones(len(TOY_X)))


def test_regressor_fits_a_linear_relationship():
    """CustomSVR replaces an `SVR` the regression path used to fall back to."""
    X, y = make_regression(n_samples=1500, n_features=10, noise=12.0, random_state=0)
    X = StandardScaler().fit_transform(X)
    y = y / y.std()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    predicted = CustomSVR(C=10.0, epsilon=0.1, learning_rate=0.05, n_iters=2000).fit(X_train, y_train).predict(X_test)

    residual = np.sum((y_test - predicted) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    assert 1 - residual > 0.9
