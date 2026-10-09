"""Tests for the LLM planning layer: planner, parser, validator, merge and endpoints.

No network anywhere. The planner is driven by a fake client that returns canned replies,
so these tests pin what the deterministic layers do with a plan -- which is the part that
has to be right every time -- rather than what a particular model happens to say.
"""

import copy
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from config.settings import SETTINGS
from src.advisors.model_advisor import ModelAdvisor
from src.advisors.preprocessing_advisor import PreprocessingAdvisor
from src.planning.merge import merge_models, merge_preprocessing, tag
from src.planning.planner import LLMPlanner
from src.planning.profile_digest import build_planning_profile
from src.planning.schema import extract_json, parse_plan
from src.planning.service import build_advisor_plan
from src.planning.validator import validate_plan
from src.profiling.dataset_profiler import DatasetProfiler

# The action keys the Preprocessing page maps onto its selectors
# (IMPUTER_BY_ACTION / ENCODER_BY_ACTION / SCALER_BY_ACTION in PreprocessingStep.tsx).
# A planned recommendation with any other key would be shown and silently not applied.
FRONTEND_ACTIONS = {
    "imputation": {"mean", "median", "mode", "knn", "drop"},
    "encoding": {"onehot", "ordinal", "label"},
    "scaling": {"standard", "minmax", "robust", "log1p"},
}


class FakeClient:
    """Stands in for LLMClient. Returns a canned reply and records every call."""

    def __init__(self, reply=None, available=True, error=None, delay=0.0):
        self.reply = reply
        self.available = available
        self._error = error
        self.delay = delay
        self.model = "fake-planner"
        self.calls: list[dict] = []

    @property
    def is_available(self) -> bool:
        return self.available

    @property
    def last_error(self):
        return self._error

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.reply is None or isinstance(self.reply, str):
            return self.reply
        return json.dumps(self.reply)


def _frame(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    age = rng.normal(40, 12, rows).round(1)
    age[rng.choice(rows, 12, replace=False)] = np.nan            # 6% missing, numeric
    score = rng.normal(0, 1, rows)
    score[rng.choice(rows, 80, replace=False)] = np.nan          # 40% missing, numeric
    city = rng.choice(["Delhi", "Mumbai", "Pune", "Chennai"], rows).astype(object)
    city[rng.choice(rows, 10, replace=False)] = None             # 5% missing, categorical
    return pd.DataFrame({
        "age": age,
        "income": rng.lognormal(10, 1, rows).round(2),           # skewed, no missing
        "balance": rng.normal(0, 500, rows).round(2),            # goes well below -1
        "score": score,
        "city": city,
        "segment": [f"S{i % 25}" for i in range(rows)],          # 25 categories
        "zip_code": [f"Z{i % 120}" for i in range(rows)],        # high cardinality
        "churn": rng.choice([0, 1], rows, p=[0.7, 0.3]),
    })


@pytest.fixture
def profile() -> dict:
    return DatasetProfiler(_frame(), target_column="churn", problem_type="Classification").compute_profile()


def _validate(profile, preprocessing=(), features=(), models=()):
    plan, malformed = parse_plan({
        "preprocessing": list(preprocessing),
        "feature_engineering": list(features),
        "models": list(models),
    })
    assert not malformed, malformed
    return validate_plan(plan, profile, "churn")


def _step(column, technique, reason="Because the profile says so (6% missing).", confidence=0.8):
    return {"column": column, "technique": technique, "reason": reason, "confidence": confidence}


def _rejected_reason(result, fragment: str) -> str:
    reasons = [r.reason for r in result.rejected]
    assert any(fragment.lower() in r.lower() for r in reasons), reasons
    return next(r for r in reasons if fragment.lower() in r.lower())


# ── Validator: accepted output fits the UI ─────────────────────────────

def test_a_valid_plan_becomes_recommendations_the_ui_can_apply(profile):
    result = _validate(
        profile,
        preprocessing=[_step("age", "median"), _step("city", "onehot"), _step("income", "log1p")],
        models=[{"model": "XGBoost", "reason": "200 rows of mixed data.", "confidence": 0.9}],
    )

    assert not result.rejected
    for rec in result.preprocessing:
        assert rec.metadata["action"] in FRONTEND_ACTIONS[rec.category]
        assert rec.metadata["source"] == "llm"
        assert rec.why_explanation  # the Why? panel is never blank
    assert [r.metadata["model_name"] for r in result.models] == ["XGBoost"]


def test_standard_scaling_a_text_column_is_rejected(profile):
    """The example from the design: StandardScaler where there is nothing numeric to scale."""
    result = _validate(profile, preprocessing=[_step("city", "StandardScaler")])

    assert result.preprocessing == []
    assert "text column" in _rejected_reason(result, "needs numbers")


def test_familiar_spellings_are_normalised_rather_than_rejected(profile):
    result = _validate(
        profile,
        preprocessing=[_step("income", "StandardScaler"), _step("city", "OneHotEncoder")],
        models=[{"model": "RandomForestClassifier", "reason": "Mixed types.", "confidence": 0.8}],
    )

    assert {r.metadata["action"] for r in result.preprocessing} == {"standard", "onehot"}
    assert result.models[0].metadata["model_name"] == "Random Forest"


@pytest.mark.parametrize(
    "step, fragment",
    [
        (_step("churn", "standard"), "prediction target"),
        (_step("salary", "median"), "no column named"),
        (_step("age", "smote"), "not a preprocessing technique"),
        (_step("segment", "onehot"), "one-hot encoding would add 25 columns"),
        (_step("city", "mean"), "mode imputation is the one"),
        (_step("income", "median"), "no missing values"),
        (_step("balance", "log1p"), "undefined at or below -1"),
        (_step("zip_code", "ordinal"), "only dropping it is accepted"),
        (_step("age", "onehot"), "already numeric"),
        (_step("age", "median", reason=""), "no reason"),
    ],
)
def test_invalid_steps_are_rejected_with_a_reason(profile, step, fragment):
    result = _validate(profile, preprocessing=[step])

    assert result.preprocessing == []
    _rejected_reason(result, fragment)


def test_column_names_tolerate_a_change_of_case_only(profile):
    result = _validate(profile, preprocessing=[_step("AGE", "median"), _step("ag", "median")])

    assert [r.metadata["column"] for r in result.preprocessing] == ["age"]
    _rejected_reason(result, "no column named 'ag'")


def test_a_dropped_column_gets_no_other_steps(profile):
    result = _validate(profile, preprocessing=[_step("age", "standard"), _step("age", "drop")])

    assert [r.metadata["action"] for r in result.preprocessing] == ["drop"]
    _rejected_reason(result, "same plan drops 'age'")


def test_only_one_technique_per_column_per_step(profile):
    result = _validate(profile, preprocessing=[_step("income", "robust"), _step("income", "standard")])

    assert [r.metadata["action"] for r in result.preprocessing] == ["robust"]
    _rejected_reason(result, "already has a scaling step")


def test_drop_is_filed_where_the_pipeline_reads_it(profile):
    """The pipeline reads drops from the imputer config alone; filed anywhere else they
    would be shown on screen and silently not applied."""
    result = _validate(profile, preprocessing=[{**_step("zip_code", "drop"), "step": "encoding"}])

    assert result.preprocessing[0].category == "imputation"


# ── Validator: feature engineering ─────────────────────────────────────

def _feature(technique, **params):
    return {"technique": technique, "params": params, "reason": "Two features correlate at r=0.9.",
            "confidence": 0.7}


def test_a_valid_feature_step_carries_the_exact_request_fields(profile):
    result = _validate(profile, features=[_feature("pca", n_components=2), _feature("select_k_best", k=3)])

    configs = {r.metadata["technique"]: r.metadata["config"] for r in result.feature_engineering}
    assert configs["pca"] == {"pca_active": True, "pca_components": 2}
    assert configs["select_k_best"] == {"select_k_best_active": True, "select_k_best_k": 3}


@pytest.mark.parametrize(
    "feature, fragment",
    [
        (_feature("pca", n_components=40), "n_components must be between"),
        (_feature("pca", n_components=0.95), "whole number of components"),
        (_feature("pca"), "whole number of components"),
        (_feature("select_k_best", k=99), "keeping all of them selects nothing"),
        (_feature("polynomial", degree=5), "degree must be 2 or 3"),
        (_feature("low_variance", threshold=4), "between 0 and 1"),
        (_feature("autoencoder"), "not a feature-engineering technique"),
    ],
)
def test_invalid_feature_steps_are_rejected(profile, feature, fragment):
    result = _validate(profile, features=[feature])

    assert result.feature_engineering == []
    _rejected_reason(result, fragment)


def test_polynomial_expansion_is_refused_when_it_would_explode():
    rng = np.random.default_rng(1)
    wide = pd.DataFrame({f"x{i}": rng.normal(size=100) for i in range(12)})
    wide["y"] = rng.choice([0, 1], 100)
    wide_profile = DatasetProfiler(wide, target_column="y", problem_type="Classification").compute_profile()

    plan, _ = parse_plan({"feature_engineering": [_feature("polynomial", degree=2)]})
    result = validate_plan(plan, wide_profile, "y")

    assert result.feature_engineering == []
    _rejected_reason(result, "makes 90 columns")


def test_features_dropped_by_the_plan_are_not_counted(profile):
    """Six features reach feature engineering here (high-cardinality zip_code never
    does). Dropping two leaves four, so k=4 would keep everything and select nothing."""
    result = _validate(
        profile,
        preprocessing=[_step("age", "drop"), _step("score", "drop")],
        features=[_feature("select_k_best", k=4)],
    )

    assert result.feature_engineering == []
    _rejected_reason(result, "between 1 and")


# ── Validator: models ──────────────────────────────────────────────────

def _model(name, reason="Suits 200 mixed-type rows."):
    return {"model": name, "reason": reason, "confidence": 0.8}


def test_models_outside_the_task_are_rejected(profile):
    result = _validate(profile, models=[_model("Linear Regression"), _model("TabNet")])

    assert result.models == []
    _rejected_reason(result, "not available for a Classification task")


def test_duplicate_models_are_kept_once(profile):
    result = _validate(profile, models=[_model("XGBoost"), _model("XGBClassifier")])

    assert [r.metadata["model_name"] for r in result.models] == ["XGBoost"]
    _rejected_reason(result, "already in this plan")


def test_svm_on_a_large_dataset_is_refused_and_not_readded(profile):
    large = copy.deepcopy(profile)
    large["shape"] = (SETTINGS.SVM_MAX_ROWS + 1, large["shape"][1])
    plan, _ = parse_plan({"models": [_model("SVM"), _model("LightGBM")]})

    result = validate_plan(plan, large, "churn")

    assert [r.metadata["model_name"] for r in result.models] == ["LightGBM"]
    assert result.excluded_models == {"SVM"}

    merged = merge_models(result.models, ModelAdvisor().recommend(large), result.excluded_models)
    assert "SVM" not in [r.metadata["model_name"] for r in merged]


# ── Parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        '{"models": []}',
        '```json\n{"models": []}\n```',
        'Here is the plan:\n{"models": []}\nHope this helps.',
    ],
)
def test_json_is_found_however_it_is_wrapped(text):
    assert extract_json(text) == {"models": []}


def test_text_that_is_not_json_is_none():
    assert extract_json("I recommend XGBoost.") is None


def test_one_malformed_item_does_not_cost_the_rest():
    plan, malformed = parse_plan({
        "preprocessing": [
            {"column": "age", "technique": "median", "reason": "r"},
            {"technique": "median", "reason": "no column"},
        ],
        "models": ["XGBoost", {"reason": "no model name"}],
    })

    assert [s.column for s in plan.preprocessing] == ["age"]
    assert [m.model for m in plan.models] == ["XGBoost"]
    assert len(malformed) == 2


def test_a_step_over_several_columns_is_expanded():
    plan, _ = parse_plan({"preprocessing": [{"columns": ["a", "b"], "technique": "standard", "reason": "r"}]})

    assert [s.column for s in plan.preprocessing] == ["a", "b"]


def test_percentage_confidence_is_read_as_a_fraction():
    plan, _ = parse_plan({"models": [{"model": "XGBoost", "reason": "r", "confidence": 86}]})

    assert plan.models[0].confidence == pytest.approx(0.86)


# ── Planner ────────────────────────────────────────────────────────────

def test_no_key_means_unavailable_and_no_call(profile):
    client = FakeClient(available=False, error="No LLM_API_KEY configured.")

    result = LLMPlanner(client).plan(profile, "churn")

    assert result.status == "unavailable"
    assert client.calls == []


@pytest.mark.parametrize("reply, fragment", [(None, "rate limit"), ("not json at all", "not valid JSON")])
def test_a_failed_call_is_a_status_not_an_exception(profile, reply, fragment):
    client = FakeClient(reply=reply, error="LLM rate limit reached.")

    result = LLMPlanner(client).plan(profile, "churn")

    assert result.status == "failed"
    assert fragment.lower() in result.error.lower()


def test_the_request_asks_for_json_and_sends_no_row_values(profile):
    client = FakeClient(reply={"models": []})

    LLMPlanner(client).plan(profile, "churn")

    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_tokens"] == SETTINGS.LLM_PLAN_MAX_TOKENS
    payload = call["user"]
    assert "zip_code" in payload                  # column names are sent...
    assert "Z17" not in payload and "Delhi" not in payload  # ...cell values never are


def test_the_digest_is_strict_json_even_for_degenerate_columns():
    """A constant column has NaN skewness, and NaN is not valid JSON."""
    frame = _frame()
    frame["constant"] = 1.0
    prof = DatasetProfiler(frame, target_column="churn", problem_type="Classification").compute_profile()

    digest = build_planning_profile(prof, "churn")

    json.dumps(digest, allow_nan=False)
    assert all(c["name"] != "churn" for c in digest["columns"])


# ── Merge with the rules ───────────────────────────────────────────────

def test_rules_fill_every_step_the_planner_left_open(profile):
    planned = _validate(profile, preprocessing=[_step("age", "knn")]).preprocessing
    rules = PreprocessingAdvisor().recommend(profile)

    merged = merge_preprocessing(planned, rules, profile)

    by_key = {(r.metadata["column"], r.category): r for r in merged}
    assert by_key[("age", "imputation")].metadata == {"column": "age", "action": "knn", "source": "llm"}
    assert by_key[("city", "encoding")].metadata["source"] == "rule"
    assert len(by_key) == len(merged), "no (column, step) appears twice"


def test_a_planner_drop_suppresses_the_rules_for_that_column(profile):
    planned = _validate(profile, preprocessing=[_step("income", "drop")]).preprocessing

    merged = merge_preprocessing(planned, PreprocessingAdvisor().recommend(profile), profile)

    assert [r.metadata["action"] for r in merged if r.metadata.get("column") == "income"] == ["drop"]


def test_a_column_the_planner_kept_is_not_dropped_by_the_rules(profile):
    """score is 40% missing, so the rules would drop it. The planner scaled it instead --
    a decision to keep it -- so the gap is filled with a plain imputer, not a drop."""
    planned = _validate(profile, preprocessing=[_step("score", "robust")]).preprocessing

    merged = merge_preprocessing(planned, PreprocessingAdvisor().recommend(profile), profile)

    actions = {r.metadata["action"] for r in merged if r.metadata.get("column") == "score"}
    assert actions == {"robust", "median"}


def test_a_short_model_list_is_topped_up_from_the_rules(profile):
    planned = _validate(profile, models=[_model("CatBoost")]).models

    merged = merge_models(planned, ModelAdvisor().recommend(profile), set())

    names = [r.metadata["model_name"] for r in merged]
    assert names[0] == "CatBoost" and len(names) == SETTINGS.MIN_RECOMMENDED_MODELS
    assert [r.metadata["source"] for r in merged][1:] == ["rule"] * (len(names) - 1)


def test_no_planned_models_means_the_rule_ranking_unchanged(profile):
    rules = ModelAdvisor().recommend(profile)

    merged = merge_models([], tag(ModelAdvisor().recommend(profile), "rule"), set())

    assert [r.metadata["model_name"] for r in merged] == [
        r.metadata["model_name"] for r in sorted(rules, key=lambda r: r.confidence_score, reverse=True)
    ]


# ── End to end through the API ─────────────────────────────────────────

PLAN = {
    "preprocessing": [
        _step("age", "knn", reason="6% missing in a column related to others."),
        _step("city", "StandardScaler", reason="Wrong on purpose."),
        _step("zip_code", "drop", reason="120 distinct values: an identifier."),
    ],
    "feature_engineering": [_feature("select_k_best", k=3)],
    "models": [_model("LightGBM"), _model("Logistic Regression"), _model("XGBoost")],
}


@pytest.fixture
def api(monkeypatch):
    """A session at the analysis step, with the planner wired to a fake client."""
    from fastapi.testclient import TestClient

    from backend.api import advisors
    from backend.main import app

    fake = FakeClient(reply=PLAN)
    monkeypatch.setattr(
        advisors, "build_advisor_plan",
        lambda profile, target: build_advisor_plan(profile, target, client=fake),
    )

    client = TestClient(app)
    headers = {"X-Session-Id": client.post("/api/session").json()["session_id"]}
    client.post("/api/datasets", headers=headers,
                files={"file": ("d.csv", _frame().to_csv(index=False), "text/csv")}).raise_for_status()
    client.post("/api/datasets/target", headers=headers,
                json={"target_column": "churn", "problem_type": "Classification"}).raise_for_status()
    client.get("/api/datasets/profile", headers=headers).raise_for_status()
    return client, headers, fake


def test_one_planning_call_serves_every_advisor(api):
    client, headers, fake = api

    pre = client.get("/api/advisors/preprocessing", headers=headers).json()
    models = client.get("/api/advisors/model", headers=headers).json()
    features = client.get("/api/advisors/features", headers=headers).json()

    assert len(fake.calls) == 1
    assert pre["planner"]["status"] == "ok"
    sources = {(r["metadata"]["column"], r["category"]): r["metadata"]["source"] for r in pre["recommendations"]}
    assert sources[("age", "imputation")] == "llm"
    assert sources[("city", "encoding")] == "rule"
    assert [r["item"] for r in pre["planner"]["rejected"]] == ["standard on 'city'"]

    assert [r["metadata"]["model_name"] for r in models["recommendations"]] == [
        "LightGBM", "Logistic Regression", "XGBoost"
    ]
    assert features["recommendations"][0]["metadata"]["config"]["select_k_best_k"] == 3


def test_without_a_key_the_advisors_are_exactly_the_rules(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    headers = {"X-Session-Id": client.post("/api/session").json()["session_id"]}
    client.post("/api/datasets", headers=headers,
                files={"file": ("d.csv", _frame().to_csv(index=False), "text/csv")}).raise_for_status()
    client.post("/api/datasets/target", headers=headers,
                json={"target_column": "churn", "problem_type": "Classification"}).raise_for_status()
    client.get("/api/datasets/profile", headers=headers).raise_for_status()

    pre = client.get("/api/advisors/preprocessing", headers=headers).json()
    models = client.get("/api/advisors/model", headers=headers).json()
    features = client.get("/api/advisors/features", headers=headers).json()

    assert pre["planner"]["status"] == "unavailable"
    assert {r["metadata"]["source"] for r in pre["recommendations"]} == {"rule"}
    assert len(models["recommendations"]) == len(ModelAdvisor().recommend(
        DatasetProfiler(_frame(), target_column="churn", problem_type="Classification").compute_profile()
    ))
    assert features["recommendations"] == []


def test_choosing_a_new_target_plans_again(api):
    client, headers, fake = api
    client.get("/api/advisors/preprocessing", headers=headers)

    client.post("/api/datasets/target", headers=headers,
                json={"target_column": "churn", "problem_type": "Classification"}).raise_for_status()
    client.get("/api/datasets/profile", headers=headers).raise_for_status()
    client.get("/api/advisors/preprocessing", headers=headers)

    assert len(fake.calls) == 2


def test_concurrent_first_requests_share_one_call(profile):
    """Both advisor pages can be requested at once; only one may pay for the plan."""
    from backend.api import advisors
    from backend.core.session import Session

    fake = FakeClient(reply={"models": [_model("XGBoost")]}, delay=0.3)
    session = Session("c" * 32)
    session.set("profile", profile)
    session.set("target_column", "churn")

    original = advisors.build_advisor_plan
    advisors.build_advisor_plan = lambda p, t: build_advisor_plan(p, t, client=fake)
    try:
        threads = [threading.Thread(target=advisors._plan, args=(session,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        advisors.build_advisor_plan = original

    assert len(fake.calls) == 1
    assert session.get("ai_plan").status == "ok"
