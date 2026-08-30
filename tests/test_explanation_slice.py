"""Unit tests for reducing SHAP output to a plottable single-sample slice.

`shap.plots.waterfall` accepts exactly one thing: an Explanation whose values are
one-dimensional. Every other shape raises, and the other shapes are the common case --
a binary tree model returns (samples, features, 2), multiclass returns
(samples, features, n_classes), and the Kernel fallback returns a bare ndarray with no
Explanation wrapper. These tests pin each of those down.
"""

import numpy as np
import pytest
import shap

from src.explainability.explainer import select_explanation_slice

FEATURES = ["a", "b", "c"]


def _is_plottable(sliced) -> bool:
    """True when shap.plots.waterfall would accept this object."""
    return sliced is not None and np.asarray(sliced.values).ndim == 1


def test_two_dimensional_array_slices_to_one_sample():
    """The regression / single-output case: (samples, features)."""
    values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    sliced = select_explanation_slice(values, 1, feature_names=FEATURES)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([0.4, 0.5, 0.6])


def test_three_dimensional_array_selects_the_requested_class():
    """Binary and multiclass tree models return (samples, features, classes)."""
    values = np.zeros((4, 3, 2))
    values[1, :, 0] = [1.0, 2.0, 3.0]
    values[1, :, 1] = [9.0, 8.0, 7.0]

    sliced = select_explanation_slice(values, 1, class_idx=1, feature_names=FEATURES)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([9.0, 8.0, 7.0])


def test_three_dimensional_defaults_to_first_class_when_index_out_of_range():
    """An out-of-range class must not raise; it falls back to the first."""
    values = np.zeros((2, 3, 2))
    values[0, :, 0] = [1.0, 1.0, 1.0]

    sliced = select_explanation_slice(values, 0, class_idx=99, feature_names=FEATURES)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([1.0, 1.0, 1.0])


def test_per_class_list_is_reduced():
    """Older SHAP versions return one array per class."""
    values = [np.ones((3, 3)), np.full((3, 3), 5.0)]
    sliced = select_explanation_slice(values, 0, class_idx=1, feature_names=FEATURES)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([5.0, 5.0, 5.0])


def test_explanation_object_is_unwrapped_and_sliced():
    """An Explanation carrying its own base values and data is handled directly."""
    explanation = shap.Explanation(
        values=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        base_values=np.array([0.25, 0.75]),
        data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        feature_names=FEATURES,
    )
    sliced = select_explanation_slice(explanation, 1)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([0.4, 0.5, 0.6])
    assert sliced.base_values == pytest.approx(0.75)


def test_one_dimensional_input_passes_through():
    """A single sample's contributions need no slicing."""
    sliced = select_explanation_slice(np.array([0.1, 0.2, 0.3]), 0, feature_names=FEATURES)

    assert _is_plottable(sliced)
    assert list(sliced.values) == pytest.approx([0.1, 0.2, 0.3])


# ── Base values ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "base_value, class_idx, expected",
    [
        (0.5, None, 0.5),                       # scalar
        (np.array([0.2, 0.8]), 1, 0.8),         # one per class
        (np.array([[0.1, 0.9]]), 1, 0.9),       # per sample, per class
        (None, None, 0.0),                      # absent
    ],
)
def test_base_value_is_reduced_to_a_scalar(base_value, class_idx, expected):
    """Waterfall requires a scalar baseline; multiclass explainers supply arrays."""
    values = np.zeros((1, 3, 2))
    sliced = select_explanation_slice(
        values, 0, class_idx=class_idx, feature_names=FEATURES, base_value=base_value
    )
    assert sliced.base_values == pytest.approx(expected)


# ── Graceful failure ──────────────────────────────────────────────────


def test_out_of_range_sample_returns_none():
    """A slider index beyond the explained subset yields None, not an exception."""
    assert select_explanation_slice(np.zeros((3, 3)), 99, feature_names=FEATURES) is None


def test_unusable_input_returns_none():
    """Anything that cannot be coerced yields None so the caller falls back."""
    assert select_explanation_slice("not an array", 0) is None
    assert select_explanation_slice([], 0) is None


def test_mismatched_feature_values_are_dropped_not_fatal():
    """Feature data of the wrong width is omitted rather than raising."""
    sliced = select_explanation_slice(
        np.zeros((2, 3)),
        0,
        feature_names=FEATURES,
        feature_values=np.zeros((2, 7)),  # wrong width
    )
    assert _is_plottable(sliced)
    assert sliced.data is None
