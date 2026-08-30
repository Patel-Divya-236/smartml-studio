"""Unit tests for the explanation-layer additions.

Three properties matter here and are each pinned by a test:

1. SHAP contributions are only ever described in units the explainer actually produced.
   A reader told "+0.31" means "+31 percentage points" is misled whenever the space is
   log-odds, so `infer_output_space` must not claim probability for a boosted model.
2. Relative magnitudes are precomputed, never left to the language model to calculate.
3. The privacy boundary still holds for the new payload builders: aggregate statistics
   and advisor decisions leave the machine; dataset rows do not.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.advisors.base import Recommendation
from src.explainability.explainer import (
    aggregate_global_importance,
    infer_output_space,
)
from src.llm.context import (
    build_column_payload,
    build_global_payload,
    build_prediction_payload,
    build_recommendation_payload,
)


# ── Output space ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, explainer_type, expected_fragment",
    [
        (RandomForestClassifier(), "Tree", "probability"),
        (LogisticRegression(), "Linear", "log-odds"),
        (LinearRegression(), "Kernel", "target units"),
        # KernelExplainer is built on model.predict, so it explains the class index.
        (LogisticRegression(), "Kernel", "class index"),
    ],
)
def test_infer_output_space(model, explainer_type, expected_fragment):
    """Each explainer/model pairing reports the units it actually produces."""
    assert expected_fragment in infer_output_space(model, explainer_type)


def test_boosted_classifier_is_not_described_as_probability():
    """A log-odds space must never be labelled probability -- that is the units bug."""

    class XGBClassifier:
        """Minimal stand-in exposing the classifier duck-type."""

        def predict_proba(self, X):
            raise NotImplementedError

    space = infer_output_space(XGBClassifier(), "Tree")
    assert "log-odds" in space
    assert "NOT probability" in space


# ── Global importance aggregation ─────────────────────────────────────


def test_aggregate_global_importance_ranks_by_mean_absolute_value():
    """Direction is ignored; a large negative contribution still ranks highly."""
    values = np.array([[0.5, -0.9, 0.1], [0.3, -0.7, -0.05]])
    ranked = aggregate_global_importance(values, ["a", "b", "c"])

    assert [name for name, _ in ranked] == ["b", "a", "c"]
    assert ranked[0][1] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "values",
    [
        np.random.rand(4, 3),                          # (samples, features)
        np.random.rand(4, 3, 2),                       # (samples, features, classes)
        [np.random.rand(4, 3), np.random.rand(4, 3)],  # per-class list
    ],
)
def test_aggregate_global_importance_handles_shap_output_shapes(values):
    """SHAP returns several shapes; all reduce to one score per feature."""
    ranked = aggregate_global_importance(values, ["a", "b", "c"])
    assert ranked is not None
    assert len(ranked) == 3


def test_aggregate_global_importance_returns_none_when_unusable():
    """An unreadable shape yields None so the caller leaves the plot untouched."""
    assert aggregate_global_importance("not an array", ["a"]) is None
    assert aggregate_global_importance([], ["a"]) is None


def test_aggregate_global_importance_respects_top_n():
    """Only the requested number of features is returned."""
    values = np.random.rand(5, 8)
    names = [f"f{i}" for i in range(8)]
    assert len(aggregate_global_importance(values, names, top_n=3)) == 3


# ── Prediction payload: units and precomputed ratios ──────────────────


def test_prediction_payload_states_contribution_units():
    """The units travel with the numbers, so the model cannot invent a scale."""
    payload = build_prediction_payload(
        feature_names=["a", "b"],
        feature_values=np.array([1.0, 2.0]),
        shap_values=np.array([0.4, -0.2]),
        predicted_label="Churn",
        output_space="log-odds (NOT probability)",
    )
    assert "Contribution units: log-odds (NOT probability)" in payload


def test_prediction_payload_defaults_units_to_unspecified():
    """Omitting the space must not silently imply probability."""
    payload = build_prediction_payload(
        feature_names=["a"],
        feature_values=np.array([1.0]),
        shap_values=np.array([0.4]),
        predicted_label=1,
    )
    assert "unspecified model output units" in payload
    assert "probability" not in payload


def test_prediction_payload_precomputes_relative_magnitudes():
    """Ratios are computed here, not by the model -- see the arithmetic slip they fix."""
    payload = build_prediction_payload(
        feature_names=["big", "half", "tenth"],
        feature_values=np.array([1.0, 2.0, 3.0]),
        shap_values=np.array([0.40, 0.20, 0.04]),
        predicted_label=1,
    )
    assert "1.00x" in payload
    assert "0.50x" in payload
    assert "0.10x" in payload


# ── Global payload ────────────────────────────────────────────────────


def test_global_payload_precomputes_shares():
    """Shares are supplied so the model reads them rather than dividing."""
    payload = build_global_payload(
        model_name="Random Forest",
        ranked_importance=[("a", 0.75), ("b", 0.25)],
        problem_type="Classification",
        n_samples=100,
        output_space="probability",
    )
    assert "75.0%" in payload
    assert "25.0%" in payload
    assert "Random Forest" in payload


def test_global_payload_handles_no_importances():
    """An empty ranking produces a statement, not a malformed table."""
    payload = build_global_payload(
        model_name="SVM",
        ranked_importance=[],
        problem_type="Classification",
        n_samples=0,
    )
    assert "No global feature importances" in payload


# ── Advisor payloads: content and privacy ─────────────────────────────


@pytest.fixture
def sample_recommendation() -> Recommendation:
    """A representative imputation recommendation."""
    return Recommendation(
        label="Median Imputer for: salary",
        confidence_score=0.9,
        reason="Column salary has 12.0% missing values.",
        why_explanation="Median resists outliers.",
        category="imputation",
        metadata={"column": "salary", "action": "median"},
    )


@pytest.fixture
def sample_profile() -> dict:
    """A profile carrying only aggregates -- no row values."""
    return {
        "shape": (500, 6),
        "problem_type": "Classification",
        "dtypes": {"salary": "float64"},
        "missing_pct": {"salary": 12.0},
        "cardinality": {"salary": 431},
        "skewness": {"salary": 2.4},
        "outliers": {"salary": {"count": 17}},
    }


def test_recommendation_payload_carries_the_decision_and_its_basis(
    sample_recommendation, sample_profile
):
    """The rule's decision and the statistics behind it both reach the prompt."""
    payload = build_recommendation_payload(
        label=sample_recommendation.label,
        category=sample_recommendation.category,
        reason=sample_recommendation.reason,
        why_explanation=sample_recommendation.why_explanation,
        confidence_score=sample_recommendation.confidence_score,
        metadata=sample_recommendation.metadata,
        profile=sample_profile,
    )
    assert "Median Imputer for: salary" in payload
    assert "12" in payload
    assert "500 rows" in payload
    assert "do not revise it" in payload


def test_column_payload_groups_every_step_for_one_column(sample_profile):
    """Steps applied together are explained together."""
    recs = [
        Recommendation(
            label="Median Imputer for: salary",
            confidence_score=0.9,
            reason="12.0% missing.",
            why_explanation="Median resists outliers.",
            category="imputation",
            metadata={"column": "salary", "action": "median"},
        ),
        Recommendation(
            label="Log1p Transform for: salary",
            confidence_score=0.75,
            reason="Skewness 2.4 exceeds the threshold.",
            why_explanation="Compresses a long tail.",
            category="scaling",
            metadata={"column": "salary", "action": "log1p"},
        ),
    ]
    payload = build_column_payload(column="salary", recommendations=recs, profile=sample_profile)

    assert "Median Imputer" in payload
    assert "Log1p Transform" in payload
    assert "action: median" in payload
    assert "action: log1p" in payload
    assert "500 rows" in payload


def test_advisor_payloads_leak_no_row_values(sample_recommendation):
    """The privacy boundary holds for the new builders too.

    A profile is aggregates by construction, but the test pins the property so a future
    change that starts threading raw rows through fails loudly here.
    """
    sentinel = 987654.321
    profile = {
        "shape": (5, 2),
        "problem_type": "Classification",
        "dtypes": {"salary": "float64"},
        "missing_pct": {"salary": 0.0},
        "cardinality": {"salary": 5},
        "raw_rows": [[sentinel, 1], [sentinel, 0]],  # must never be read
    }

    rec_payload = build_recommendation_payload(
        label=sample_recommendation.label,
        category=sample_recommendation.category,
        reason=sample_recommendation.reason,
        why_explanation=sample_recommendation.why_explanation,
        confidence_score=sample_recommendation.confidence_score,
        metadata=sample_recommendation.metadata,
        profile=profile,
    )
    col_payload = build_column_payload(
        column="salary", recommendations=[sample_recommendation], profile=profile
    )

    assert str(sentinel) not in rec_payload
    assert str(sentinel) not in col_payload
