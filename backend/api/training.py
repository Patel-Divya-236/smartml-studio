"""Model training as a background job, with WebSocket progress and a polling fallback.

Training is the one stage long enough that a synchronous request would leave the UI with
nothing to show. `POST /jobs` returns immediately; the client watches
`WS /jobs/{id}/progress` for an event per model, and `GET /jobs/{id}` returns the same
payload for clients that cannot hold a socket open. The polling route is not decorative —
a dropped socket during a long run is the most likely source of a confusing bug, and
having a non-realtime path makes it debuggable.
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.api.deps import get_session, require
from backend.core.session import STORE, Session
from src.evaluation.metrics import compute_metrics
from src.models.model_trainer import SUPPORTED_MODELS, ModelTrainer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


class TrainRequest(BaseModel):
    """Which models to train."""

    models: list[str]


class Job:
    """One training run's live state."""

    def __init__(self, job_id: str, session_id: str, models: list[str]) -> None:
        """Create a queued job."""
        self.id = job_id
        self.session_id = session_id
        self.status = "queued"
        self.events: list[dict[str, Any]] = []
        self.total = len(models)
        self.completed = 0
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view of the job, identical over WebSocket and polling."""
        return {
            "job_id": self.id,
            "status": self.status,
            "completed": self.completed,
            "total": self.total,
            "events": self.events,
            "error": self.error,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 2),
        }


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _run_training(job: Job, session: Session, models: list[str]) -> None:
    """Train in a worker thread, appending an event per model."""
    try:
        job.status = "running"

        X_train = session.get("feature_engineered_train")
        X_test = session.get("feature_engineered_test")
        if X_train is None:
            X_train = session.get("preprocessed_train")
            X_test = session.get("preprocessed_test")

        y_train = np.asarray(session.get("y_train"))
        y_test = np.asarray(session.get("y_test"))
        X_train_arr = np.asarray(X_train)
        X_test_arr = np.asarray(X_test)

        def on_progress(event: dict[str, Any]) -> None:
            job.events.append(event)
            job.completed = event.get("completed", job.completed)

        trainer = ModelTrainer(problem_type=session.get("problem_type") or "Classification")
        results = trainer.train_models(
            X_train_arr, y_train, X_test_arr, y_test, models,
            on_progress=on_progress,
            continue_on_error=True,
        )

        if not results:
            job.status = "failed"
            job.error = "Every selected model failed to train."
            return

        session.set("X_train", X_train_arr)
        session.set("X_test", X_test_arr)
        session.set("trained_models", results)
        session.set("training_failures", trainer.failures)

        comparison = compute_metrics(results, y_test, session.get("problem_type"))
        session.set("model_comparison", comparison)

        job.status = "completed"
    except Exception as exc:
        logger.exception("Training job %s failed", job.id[:8])
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = time.time()


@router.get("/available")
def available_models(session: Session = Depends(get_session)) -> dict:
    """List the models valid for the current task type.

    Read from the trainer so the two can never disagree. Time Series shares the
    continuous-target branch, as it does everywhere else in the pipeline.
    """
    problem_type = session.get("problem_type") or "Classification"
    key = "Classification" if problem_type == "Classification" else "Regression"
    return {"models": list(SUPPORTED_MODELS[key]), "problem_type": problem_type}


@router.post("/jobs")
def start_job(payload: TrainRequest, session: Session = Depends(get_session)) -> dict:
    """Start a training run and return its job id immediately."""
    require(session, "preprocessed_train", "preprocessing")
    if not payload.models:
        raise HTTPException(status_code=422, detail="Select at least one model.")

    session.set("selected_models", payload.models)
    job = Job(uuid.uuid4().hex, session.id, payload.models)
    with _JOBS_LOCK:
        JOBS[job.id] = job

    threading.Thread(
        target=_run_training, args=(job, session, payload.models), daemon=True
    ).start()

    return job.snapshot()


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Polling fallback for clients without a live WebSocket."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job.snapshot()


@router.websocket("/jobs/{job_id}/progress")
async def job_progress(websocket: WebSocket, job_id: str) -> None:
    """Stream job state until the run finishes.

    Polls the in-process job object rather than using a queue: the run is bounded, the
    events list is small, and this keeps the socket and the polling route reading the
    exact same snapshot, so the two can never disagree.
    """
    await websocket.accept()
    job = JOBS.get(job_id)
    if job is None:
        await websocket.send_json({"status": "failed", "error": "Unknown job."})
        await websocket.close()
        return

    last_sent = -1
    try:
        while True:
            snapshot = job.snapshot()
            if len(job.events) != last_sent or job.status in {"completed", "failed"}:
                await websocket.send_json(snapshot)
                last_sent = len(job.events)
            if job.status in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        logger.info("Client disconnected from job %s.", job_id[:8])
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@router.get("/results")
def training_results(session: Session = Depends(get_session)) -> dict:
    """Summarise the trained models, including any that failed."""
    trained = require(session, "trained_models", "training")
    failures = session.get("training_failures") or {}

    return {
        "models": [
            {
                "name": name,
                "fit_time": round(float(info["fit_time"]), 4),
                "predict_time": round(float(info["predict_time"]), 4),
                "has_probabilities": info.get("y_prob") is not None,
            }
            for name, info in trained.items()
        ],
        "failures": [{"name": n, "error": e} for n, e in failures.items()],
        "completed_steps": session.completed_steps(),
    }
