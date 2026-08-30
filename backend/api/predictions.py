"""Prediction (single model or ensemble) and the exportable artifacts."""

import io
import logging
import pickle

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend.api.deps import get_session, require
from backend.core.serialization import dataframe_to_records, to_jsonable
from backend.core.session import Session
from src.ensemble.hybrid_ensemble import HybridEnsemble
from src.llm.report_narrator import ReportNarrator
from src.reporting.report import build_evaluation_report

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predictions"])


class PredictRequest(BaseModel):
    """Predict with one model, or with a weighted/majority ensemble of several."""

    mode: str = "single"                 # "single" | "ensemble"
    model: str | None = None
    ensemble_models: list[str] = []
    voting: str = "majority"             # "majority" | "weighted" | "average"
    weights: list[float] | None = None


def _decode(session: Session, values: np.ndarray) -> list:
    """Map encoded class indices back to their original labels."""
    encoder = session.get("label_encoder")
    if encoder is None:
        return to_jsonable(values)
    try:
        return [str(v) for v in encoder.inverse_transform(np.asarray(values).astype(int))]
    except Exception:
        return to_jsonable(values)


@router.post("/predictions")
def predict(payload: PredictRequest, session: Session = Depends(get_session)) -> dict:
    """Run predictions over the held-out test set and store them for export."""
    trained = require(session, "trained_models", "training")
    X_test = np.asarray(session.get("X_test"))
    y_test = np.asarray(session.get("y_test"))

    if payload.mode == "ensemble":
        chosen = [m for m in payload.ensemble_models if m in trained]
        if len(chosen) < 2:
            raise HTTPException(status_code=422, detail="Pick at least two trained models.")
        ensemble = HybridEnsemble(
            voting=payload.voting,
            weights=payload.weights,
            problem_type=session.get("problem_type") or "Classification",
        ).fit([trained[m]["instance"] for m in chosen])
        y_pred = np.asarray(ensemble.predict(X_test))
        session.set("ensemble", ensemble)
        label = f"Ensemble ({payload.voting}) of {', '.join(chosen)}"
    else:
        if not payload.model or payload.model not in trained:
            raise HTTPException(status_code=422, detail="Choose a trained model.")
        y_pred = np.asarray(trained[payload.model]["y_pred"])
        session.set("ensemble", None)
        label = payload.model

    frame = pd.DataFrame({
        "row": np.arange(len(y_pred)),
        "actual": _decode(session, y_test),
        "predicted": _decode(session, y_pred),
    })
    if session.get("problem_type") == "Classification":
        frame["correct"] = frame["actual"].astype(str) == frame["predicted"].astype(str)
        accuracy = float(frame["correct"].mean())
    else:
        frame["error"] = np.asarray(y_test, dtype=float) - np.asarray(y_pred, dtype=float)
        accuracy = None

    session.set("predictions", frame)
    session.set("selected_prediction_model", label)

    return {
        "strategy": label,
        "count": int(len(frame)),
        "accuracy": accuracy,
        "preview": dataframe_to_records(frame, limit=100),
        "completed_steps": session.completed_steps(),
    }


def _render_report(session: Session, narrate: bool) -> tuple[str, bool, str | None]:
    """Build the Markdown report. Returns (markdown, was_narrated, llm_error).

    The executive summary is an addition to the report, never a dependency of it: when no
    model is configured or the call fails, the section is omitted and the report still
    builds. The error is returned rather than raised so the caller can mention it beside
    a report that is otherwise complete.
    """
    comparison = require(session, "model_comparison", "training")
    profile = session.get("profile") or {}

    report_state = {
        "prediction_mode": session.get("selected_prediction_model"),
        "selected_model": session.get("selected_prediction_model"),
        "ensemble_voting": getattr(session.get("ensemble"), "voting", None),
        "ensemble_models": session.get("selected_models") or [],
    }

    summary: str | None = None
    llm_error: str | None = None

    if narrate:
        narrator = ReportNarrator()
        if narrator.is_available:
            try:
                summary = narrator.summarize(
                    dataset_name=session.get("dataset_name") or "dataset",
                    target_column=session.get("target_column") or "",
                    problem_type=session.get("problem_type") or "",
                    profile=profile,
                    comparison=comparison,
                    report_state=report_state,
                )
                if summary is None:
                    llm_error = narrator.client.last_error or "The model returned nothing."
            except Exception as exc:
                # A narration failure must not cost the user their report.
                logger.exception("Executive summary failed")
                llm_error = str(exc)
        else:
            llm_error = narrator.client.last_error or "No language model is configured."

    markdown = build_evaluation_report(
        dataset_name=session.get("dataset_name") or "dataset",
        target_col=session.get("target_column") or "",
        problem_type=session.get("problem_type") or "",
        profile=profile,
        model_comparison=comparison,
        report_state=report_state,
        executive_summary=summary,
    )
    return markdown, summary is not None, llm_error


@router.get("/artifacts/report/preview")
def preview_report(narrate: bool = False, session: Session = Depends(get_session)) -> dict:
    """Return the report as text so it can be read before downloading."""
    markdown, narrated, llm_error = _render_report(session, narrate)
    return {"markdown": markdown, "narrated": narrated, "llm_error": llm_error}


@router.get("/artifacts/report")
def download_report(narrate: bool = False, session: Session = Depends(get_session)) -> Response:
    """Build the Markdown evaluation report as a file download."""
    markdown, _, _ = _render_report(session, narrate)
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="smartml_report.md"'},
    )


@router.get("/artifacts/predictions.csv")
def download_predictions(session: Session = Depends(get_session)) -> StreamingResponse:
    """Export the prediction table as CSV."""
    frame: pd.DataFrame = require(session, "predictions", "prediction")
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@router.get("/artifacts/model/{model_name}")
def download_model(model_name: str, session: Session = Depends(get_session)) -> Response:
    """Export one trained estimator as a pickle."""
    trained = require(session, "trained_models", "training")
    if model_name not in trained:
        raise HTTPException(status_code=404, detail="That model was not trained.")

    payload = pickle.dumps(trained[model_name]["instance"])
    safe = model_name.lower().replace(" ", "_")
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe}.pkl"'},
    )


@router.get("/artifacts/summary")
def export_summary(session: Session = Depends(get_session)) -> dict:
    """What is available to download, so the export page can disable dead buttons."""
    trained = session.get("trained_models") or {}
    return {
        "dataset_name": session.get("dataset_name"),
        "target_column": session.get("target_column"),
        "problem_type": session.get("problem_type"),
        "models": list(trained.keys()),
        "has_predictions": session.get("predictions") is not None,
        "has_comparison": session.get("model_comparison") is not None,
        "prediction_strategy": session.get("selected_prediction_model"),
    }
