"""Unit tests for the API's JSON conversion.

The load-bearing test here is the plain-`int` one. numpy and pandas types were handled
from the start, but a plain Python `int` — which is what `Series.to_dict()` returns under
pandas 2.x — fell past every branch and was stringified by the final `str(value)` catch.
Nothing raised. Counts simply arrived in the browser as `"79"` instead of `79`, and every
chart that plotted them rendered blank.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.core.serialization import (
    dataframe_to_records,
    recommendation_to_dict,
    to_jsonable,
)
from src.advisors.base import Recommendation


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (7, int),
        (np.int64(7), int),
        (np.int32(7), int),
        (1.5, float),
        (np.float64(1.5), float),
        (True, bool),
        (np.bool_(False), bool),
        ("text", str),
        (np.str_("text"), str),
    ],
)
def test_scalars_keep_their_type(value, expected_type):
    """Numbers must stay numbers. A stringified count breaks every chart silently."""
    assert type(to_jsonable(value)) is expected_type


def test_plain_int_is_not_stringified():
    """Regression: pandas 2.x returns plain ints, which were being turned into strings."""
    counts = pd.Series(["a", "a", "b"]).value_counts().to_dict()
    converted = to_jsonable(counts)

    assert all(isinstance(v, int) for v in converted.values())
    assert converted == {"a": 2, "b": 1}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.nan])
def test_non_finite_floats_become_none(value):
    """NaN and Infinity are not valid JSON and break JSON.parse in the browser."""
    assert to_jsonable(value) is None


def test_nested_structures_are_converted_throughout():
    """Conversion has to reach values nested inside dicts and lists."""
    payload = {
        "shape": (np.int64(100), np.int64(5)),
        "counts": {"a": 3, "b": np.int64(4)},
        "scores": [np.float64(0.5), float("nan")],
    }
    converted = to_jsonable(payload)

    assert converted["shape"] == [100, 5]
    assert all(isinstance(v, int) for v in converted["counts"].values())
    assert converted["scores"] == [0.5, None]


def test_dict_keys_become_strings():
    """JSON object keys are strings; numpy label keys must not leak through."""
    converted = to_jsonable({np.int64(1): 10, "b": 20})
    assert set(converted) == {"1", "b"}


def test_dataframe_to_records_respects_the_limit():
    """Previews are capped so a large upload cannot produce a huge response."""
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    records = dataframe_to_records(df, limit=5)

    assert len(records) == 5
    assert all(isinstance(row["a"], int) for row in records)


def test_dataframe_records_convert_missing_values():
    """A NaN cell must serialise as null, not as the string 'nan'."""
    df = pd.DataFrame({"a": [1.0, np.nan]})
    records = dataframe_to_records(df)

    assert records[0]["a"] == 1.0
    assert records[1]["a"] is None


def test_recommendation_serialises_every_display_field():
    """The card renders these fields directly, so all of them must survive."""
    rec = Recommendation(
        label="Median Imputer for: age",
        confidence_score=0.9,
        reason="12% missing.",
        why_explanation="Median resists outliers.",
        category="imputation",
        metadata={"column": "age", "action": "median", "count": np.int64(3)},
    )
    payload = recommendation_to_dict(rec)

    assert payload["label"] == "Median Imputer for: age"
    # round(0.9 * 5) is round(4.5), which Python resolves to 4 (banker's rounding).
    assert payload["star_rating"] == 4
    assert isinstance(payload["confidence_score"], float)
    assert payload["metadata"]["count"] == 3
    assert isinstance(payload["metadata"]["count"], int)


def test_numpy_array_becomes_a_list_of_primitives():
    """Charts receive arrays as plain JSON lists."""
    converted = to_jsonable(np.array([1, 2, 3]))
    assert converted == [1, 2, 3]
    assert all(isinstance(v, int) for v in converted)


def test_confusion_matrix_shape_survives():
    """A 2-D integer array keeps its nesting and its numeric type."""
    converted = to_jsonable(np.array([[5, 1], [2, 7]]))
    assert converted == [[5, 1], [2, 7]]
    assert isinstance(converted[0][0], int)


def test_finite_check_does_not_reject_zero():
    """Zero is finite; an over-eager falsiness check would drop it."""
    assert to_jsonable(0) == 0
    assert to_jsonable(0.0) == 0.0
    assert not math.isnan(to_jsonable(0.0))


# ── Model catalogue ───────────────────────────────────────────────────


def test_every_advertised_model_can_actually_be_built():
    """The API must not offer a model the trainer cannot instantiate.

    Regression: the endpoint kept its own hardcoded list, which had drifted to include
    "Decision Tree" — a name `_get_model_instance` rejects. Selecting it failed at
    training time with no earlier warning.
    """
    from src.models.model_trainer import SUPPORTED_MODELS, ModelTrainer

    for task, names in SUPPORTED_MODELS.items():
        trainer = ModelTrainer(task)
        for name in names:
            trainer._get_model_instance(name)  # raises ValueError on an unknown name


def test_naive_bayes_is_classification_only():
    """It has no regression form, so it must not be advertised for one."""
    from src.models.model_trainer import SUPPORTED_MODELS

    assert "Naive Bayes" in SUPPORTED_MODELS["Classification"]
    assert "Naive Bayes" not in SUPPORTED_MODELS["Regression"]
