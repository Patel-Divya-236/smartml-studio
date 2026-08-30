"""Tests for the evaluation report endpoints.

The bug these pin: the download route called `ReportNarrator.narrate`, a method that
does not exist — the class exposes `summarize`. It returned 500 whenever narration was
requested, which is the default state of the toggle when a model is configured. It went
unnoticed because `TestClient(app)` without a context manager skips startup, so the LLM
key was never loaded and the narrator branch was never entered.

The narrator is therefore stubbed here rather than left to availability, so the branch is
always exercised.
"""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client() -> TestClient:
    """A client with a trained pipeline already in session."""
    test_client = TestClient(app)
    session_id = test_client.post("/api/session").json()["session_id"]
    headers = {"X-Session-Id": session_id}

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, 200),
        "score": rng.normal(50, 10, 200),
        "region": rng.choice(["north", "south"], 200),
        "churn": rng.choice(["yes", "no"], 200),
    })
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)

    test_client.post(
        "/api/datasets",
        files={"file": ("demo.csv", buffer.getvalue(), "text/csv")},
        headers=headers,
    )
    test_client.post("/api/datasets/target", json={"target_column": "churn"}, headers=headers)
    test_client.get("/api/datasets/profile", headers=headers)
    test_client.post(
        "/api/pipeline/preprocess",
        json={"test_size": 0.25, "impute": {}, "encode": {"region": "One-Hot"}, "scale": {}},
        headers=headers,
    )

    job = test_client.post(
        "/api/training/jobs", json={"models": ["Random Forest"]}, headers=headers
    ).json()["job_id"]

    for _ in range(200):
        status = test_client.get(f"/api/training/jobs/{job}").json()
        if status["status"] in {"completed", "failed"}:
            break
    assert status["status"] == "completed", status

    test_client.headers.update(headers)
    return test_client


def test_report_downloads_without_narration(client):
    """The plain report needs no language model at all."""
    response = client.get("/api/artifacts/report?narrate=false")

    assert response.status_code == 200
    assert "SmartML Studio" in response.text
    assert "attachment" in response.headers["content-disposition"]


def test_report_with_narration_requested_still_returns_200(client, monkeypatch):
    """Regression: this returned 500 because `narrate` is not a method on the narrator."""
    from backend.api import predictions

    class StubNarrator:
        is_available = True
        client = type("C", (), {"last_error": None})()

        def summarize(self, **_kwargs):
            return "This run trained one model on a small, balanced dataset."

    monkeypatch.setattr(predictions, "ReportNarrator", StubNarrator)
    response = client.get("/api/artifacts/report?narrate=true")

    assert response.status_code == 200
    assert "This run trained one model" in response.text


def test_narration_failure_does_not_lose_the_report(client, monkeypatch):
    """A broken narrator costs the summary, never the report."""
    from backend.api import predictions

    class ExplodingNarrator:
        is_available = True
        client = type("C", (), {"last_error": None})()

        def summarize(self, **_kwargs):
            raise RuntimeError("provider unreachable")

    monkeypatch.setattr(predictions, "ReportNarrator", ExplodingNarrator)
    response = client.get("/api/artifacts/report/preview?narrate=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["narrated"] is False
    assert "provider unreachable" in payload["llm_error"]
    assert "SmartML Studio" in payload["markdown"]


def test_preview_returns_the_same_text_as_the_download(client):
    """The preview has to be the file, not an approximation of it."""
    preview = client.get("/api/artifacts/report/preview?narrate=false").json()
    download = client.get("/api/artifacts/report?narrate=false")

    assert preview["markdown"] == download.text
    assert preview["narrated"] is False


def test_preview_contains_the_metrics_table(client):
    """The comparison table is the substance of the report."""
    markdown = client.get("/api/artifacts/report/preview?narrate=false").json()["markdown"]

    assert "| Model Name |" in markdown
    assert "Random Forest" in markdown


def test_report_requires_training_first(client):
    """A session with no trained models gets a clear 409, not a stack trace."""
    fresh = TestClient(app)
    session_id = fresh.post("/api/session").json()["session_id"]

    response = fresh.get("/api/artifacts/report", headers={"X-Session-Id": session_id})

    assert response.status_code == 409
    assert "training" in response.json()["detail"].lower()
