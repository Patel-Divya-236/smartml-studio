"""Model comparison, diagnostics, and SHAP explainability.

Every number served here is computed locally by sklearn or SHAP. The LLM routes at the
bottom receive those numbers already computed and only rewrite them as prose — they never
produce a metric, an attribution, or a ranking.
"""

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix, roc_curve

from backend.api.deps import get_session, require
from backend.core.serialization import to_jsonable
from backend.core.session import Session
from src.evaluation.metrics import extract_feature_importance
from src.explainability.explainer import (
    ModelExplainer,
    aggregate_global_importance,
    extract_sample_contributions,
    infer_output_space,
)
from src.llm.global_narrator import GlobalNarrator
from src.llm.prediction_narrator import PredictionNarrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class NarrateSampleRequest(BaseModel):
    """Which model and test row to explain in prose."""

    model: str
    sample_index: int = 0


@router.get("/comparison")
def comparison(session: Session = Depends(get_session)) -> dict:
    """Return the metrics table plus the best model per primary metric."""
    df: pd.DataFrame = require(session, "model_comparison", "training")
    problem_type = session.get("problem_type")

    primary = "Accuracy" if problem_type == "Classification" else "R² Score"
    best = None
    if primary in df.columns and not df.empty:
        best = str(df.loc[df[primary].idxmax(), "Model Name"])

    return {
        "columns": [str(c) for c in df.columns],
        "rows": to_jsonable(df),
        "primary_metric": primary,
        "best_model": best,
        "problem_type": problem_type,
        "completed_steps": session.completed_steps(),
    }


@router.get("/feature-importance/{model_name}")
def feature_importance(model_name: str, session: Session = Depends(get_session)) -> dict:
    """Built-in feature importance for one model, when the estimator exposes it."""
    trained = require(session, "trained_models", "training")
    if model_name not in trained:
        raise HTTPException(status_code=404, detail="That model was not trained.")

    names = [str(n) for n in (session.get("feature_names") or [])]
    df = extract_feature_importance(trained[model_name]["instance"], names)
    if df is None:
        return {"available": False, "importances": []}

    return {
        "available": True,
        "importances": [
            {"feature": str(r["Feature"]), "importance": to_jsonable(r["Importance"])}
            for _, r in df.head(25).iterrows()
        ],
    }


@router.get("/diagnostics/{model_name}")
def diagnostics(model_name: str, session: Session = Depends(get_session)) -> dict:
    """Confusion matrix and ROC for classification; actual-vs-predicted for regression."""
    trained = require(session, "trained_models", "training")
    if model_name not in trained:
        raise HTTPException(status_code=404, detail="That model was not trained.")

    info = trained[model_name]
    y_test = np.asarray(session.get("y_test"))
    y_pred = np.asarray(info["y_pred"])
    problem_type = session.get("problem_type")

    if problem_type != "Classification":
        residuals = y_test - y_pred
        return {
            "problem_type": problem_type,
            "actual_vs_predicted": {
                "actual": to_jsonable(y_test),
                "predicted": to_jsonable(y_pred),
            },
            "residuals": {"predicted": to_jsonable(y_pred), "residual": to_jsonable(residuals)},
        }

    encoder = session.get("label_encoder")
    labels = [str(c) for c in encoder.classes_] if encoder is not None else [
        str(v) for v in np.unique(y_test)
    ]
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))

    roc = None
    y_prob = info.get("y_prob")
    if y_prob is not None and len(np.unique(y_test)) == 2:
        try:
            prob_pos = y_prob[:, 1] if np.asarray(y_prob).ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_test, prob_pos)
            roc = {"fpr": to_jsonable(fpr), "tpr": to_jsonable(tpr)}
        except Exception as exc:
            logger.debug("ROC unavailable for %s: %s", model_name, exc)

    return {
        "problem_type": problem_type,
        "labels": labels,
        "confusion_matrix": to_jsonable(matrix),
        "roc": roc,
    }


def _explanation_for(session: Session, model_name: str) -> dict:
    """Compute (and cache per model) the SHAP explanation for the test set."""
    trained = require(session, "trained_models", "training")
    if model_name not in trained:
        raise HTTPException(status_code=404, detail="That model was not trained.")

    cache = session.get("explainability_results") or {}
    if model_name in cache:
        return cache[model_name]

    X_train = np.asarray(session.get("X_train"))
    X_test = np.asarray(session.get("X_test"))
    names = [str(n) for n in (session.get("feature_names") or [])]

    try:
        explanation = ModelExplainer().explain(
            trained[model_name]["instance"], X_train, X_test, names
        )
    except Exception as exc:
        logger.exception("SHAP failed for %s", model_name)
        raise HTTPException(status_code=422, detail=f"SHAP explanation failed: {exc}") from exc

    explanation["output_space"] = infer_output_space(
        trained[model_name]["instance"], explanation["explainer_type"]
    )
    cache[model_name] = explanation
    session.set("explainability_results", cache)
    return explanation


@router.get("/shap/{model_name}/global")
def shap_global(model_name: str, session: Session = Depends(get_session)) -> dict:
    """Mean absolute SHAP per feature — what the beeswarm plot encodes as spread."""
    explanation = _explanation_for(session, model_name)
    names = [str(n) for n in (session.get("feature_names") or [])]
    ranked = aggregate_global_importance(explanation["shap_values"], names, top_n=20)

    if not ranked:
        return {"available": False, "importances": [], "output_space": explanation.get("output_space")}

    total = sum(v for _, v in ranked) or 1.0
    return {
        "available": True,
        "explainer_type": explanation["explainer_type"],
        "output_space": explanation.get("output_space"),
        "is_subset": bool(explanation.get("is_subset")),
        "importances": [
            {"feature": n, "value": to_jsonable(v), "share": round(v / total * 100, 2)}
            for n, v in ranked
        ],
    }


@router.get("/shap/{model_name}/local/{sample_index}")
def shap_local(model_name: str, sample_index: int, session: Session = Depends(get_session)) -> dict:
    """Per-feature contributions for one test row, for the waterfall chart."""
    explanation = _explanation_for(session, model_name)
    contributions = extract_sample_contributions(explanation["shap_values"], sample_index)
    if contributions is None:
        return {"available": False, "contributions": []}

    names = [str(n) for n in (session.get("feature_names") or [])]
    X_test = np.asarray(session.get("X_test"))
    y_test = np.asarray(session.get("y_test"))
    y_pred = np.asarray(session.get("trained_models")[model_name]["y_pred"])
    encoder = session.get("label_encoder")

    def label_of(value):
        if encoder is None:
            return to_jsonable(value)
        try:
            return str(encoder.inverse_transform([int(value)])[0])
        except Exception:
            return to_jsonable(value)

    values = np.asarray(contributions, dtype=float).ravel()
    row = X_test[sample_index] if sample_index < len(X_test) else np.zeros(len(values))
    n = min(len(names), len(values), len(row))

    items = [
        {
            "feature": names[i],
            "value": to_jsonable(row[i]),
            "contribution": to_jsonable(values[i]),
        }
        for i in range(n)
    ]
    items.sort(key=lambda item: abs(item["contribution"] or 0), reverse=True)

    return {
        "available": True,
        "output_space": explanation.get("output_space"),
        "base_value": to_jsonable(np.asarray(explanation.get("base_value")).ravel()[0])
        if explanation.get("base_value") is not None else None,
        "predicted": label_of(y_pred[sample_index]) if sample_index < len(y_pred) else None,
        "actual": label_of(y_test[sample_index]) if sample_index < len(y_test) else None,
        "max_index": int(len(values) and len(y_pred) - 1),
        "contributions": items,
    }


@router.post("/shap/narrate-global")
def narrate_global(payload: NarrateSampleRequest, session: Session = Depends(get_session)) -> dict:
    """Describe, in prose, which features the model relies on overall."""
    narrator = GlobalNarrator()
    if not narrator.is_available:
        return {"narrative": None, "available": False, "error": narrator.client.last_error}

    explanation = _explanation_for(session, payload.model)
    names = [str(n) for n in (session.get("feature_names") or [])]
    ranked = aggregate_global_importance(explanation["shap_values"], names, top_n=20)
    if not ranked:
        return {"narrative": None, "available": True, "error": "No usable importances."}

    narrative = narrator.narrate(
        model_name=payload.model,
        ranked_importance=ranked,
        problem_type=session.get("problem_type") or "Classification",
        n_samples=int(len(np.asarray(session.get("X_test")))),
        output_space=explanation.get("output_space"),
    )
    return {
        "narrative": narrative,
        "available": True,
        "error": None if narrative else narrator.client.last_error,
    }


@router.post("/shap/narrate-local")
def narrate_local(payload: NarrateSampleRequest, session: Session = Depends(get_session)) -> dict:
    """Explain one prediction in plain English from its SHAP attributions."""
    narrator = PredictionNarrator()
    if not narrator.is_available:
        return {"narrative": None, "available": False, "error": narrator.client.last_error}

    explanation = _explanation_for(session, payload.model)
    contributions = extract_sample_contributions(explanation["shap_values"], payload.sample_index)
    if contributions is None:
        return {"narrative": None, "available": True, "error": "No usable attributions."}

    X_test = np.asarray(session.get("X_test"))
    y_test = np.asarray(session.get("y_test"))
    y_pred = np.asarray(session.get("trained_models")[payload.model]["y_pred"])
    encoder = session.get("label_encoder")

    def label_of(value):
        if encoder is None:
            return value
        try:
            return str(encoder.inverse_transform([int(value)])[0])
        except Exception:
            return value

    narrative = narrator.narrate(
        feature_names=[str(n) for n in (session.get("feature_names") or [])],
        feature_values=X_test[payload.sample_index],
        shap_values=contributions,
        predicted_label=label_of(y_pred[payload.sample_index]),
        actual_label=label_of(y_test[payload.sample_index]),
        problem_type=session.get("problem_type") or "Classification",
        base_value=explanation.get("base_value"),
        output_space=explanation.get("output_space"),
    )
    return {
        "narrative": narrative,
        "available": True,
        "error": None if narrative else narrator.client.last_error,
    }
