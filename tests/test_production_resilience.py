"""Regression tests for the four defects that made the deployed app unusable.

Each one was visible to users rather than theoretical:

1. Uploading a large file froze every other request, because the parse ran on the event
   loop. The stalled health probe made the platform restart the process.
2. A restart emptied the in-memory session store, so the browser's still-valid session id
   resolved to nothing and the UI demanded the dataset be uploaded again.
3. Numeric columns containing stray tokens were read as text and then treated as
   categories with tens of thousands of levels, exhausting memory during preprocessing.
4. Users saw "LLM returned an empty response". The configured model is a reasoning model
   that can put its answer outside `content` and spends part of the token budget thinking
   before it writes, so an empty `content` is a reachable state rather than a hard error.
   `_parse` now falls back to the other fields instead of reporting nothing at all.
"""

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from backend.api.datasets import _coerce_numeric_like
from backend.core import persistence
from backend.core.session import SessionStore
from src.advisors.preprocessing_advisor import PreprocessingAdvisor
from src.llm.client import LLMClient
from src.profiling.dataset_profiler import HIGH_CARDINALITY_LIMIT, DatasetProfiler


@pytest.fixture
def store(tmp_path, monkeypatch) -> SessionStore:
    """A session store whose checkpoints land in a temporary directory."""
    monkeypatch.setattr(persistence, "SESSION_DIR", tmp_path / "sessions")
    return SessionStore()


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """A frame shaped like the air-quality upload that broke in production."""
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        # numeric readings carrying a token pandas does not treat as null
        "PM10": [str(round(v, 2)) if i % 50 else "-" for i, v in enumerate(rng.normal(80, 20, n))],
        "CO2": [np.nan] * n,                                       # entirely missing
        "From Date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        "City": rng.choice(["Delhi", "Mumbai"], n),                # a genuine category
        "AQI": rng.normal(150, 40, n),
    })


# -- 1. The upload parse must not block the event loop ------------------------

def test_upload_parse_runs_off_the_event_loop(monkeypatch):
    """A slow parse must not delay concurrent requests.

    The parse is replaced with a blocking sleep of a known length. If it runs inline on
    the loop, the concurrent health check cannot finish until the sleep is over.
    """
    import httpx

    from backend import main as backend_main
    from backend.api import datasets

    block_seconds = 1.5

    def slow_parse(raw: bytes, name: str):
        time.sleep(block_seconds)  # blocking, never yields - like pandas holding the GIL
        return pd.DataFrame({"a": [1, 2, 3]}), []

    monkeypatch.setattr(datasets, "_parse_upload", slow_parse)

    async def scenario() -> float:
        transport = httpx.ASGITransport(app=backend_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = (await client.post("/api/session")).json()["session_id"]
            origin = time.perf_counter()

            async def upload():
                return await client.post(
                    "/api/datasets",
                    headers={"X-Session-Id": session_id},
                    files={"file": ("x.csv", b"a\n1\n", "text/csv")},
                    timeout=30,
                )

            async def health() -> float:
                await asyncio.sleep(0.2)  # let the upload reach the parse
                response = await client.get("/api/health", timeout=30)
                assert response.status_code == 200
                return time.perf_counter() - origin

            upload_response, health_at = await asyncio.gather(upload(), health())
            assert upload_response.status_code == 200
            return health_at

    health_at = asyncio.run(scenario())
    assert health_at < block_seconds * 0.8, (
        f"health waited {health_at:.2f}s for a {block_seconds}s parse - the loop is blocked"
    )


# -- 1b. Gzipped uploads ------------------------------------------------------

def _client_with_session():
    """A TestClient plus the headers for a fresh session."""
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    return client, {"X-Session-Id": client.post("/api/session").json()["session_id"]}


def _csv_bytes(rows: int = 200) -> bytes:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, rows),
        "score": rng.normal(50, 10, rows).round(3),
        "churn": rng.choice(["yes", "no"], rows),
    })
    return frame.to_csv(index=False).encode()


def test_gzipped_upload_is_read_like_a_plain_one():
    """Compression is a transport detail; the parsed result must be identical."""
    import gzip

    raw = _csv_bytes()
    client, headers = _client_with_session()

    plain = client.post("/api/datasets", headers=headers,
                        files={"file": ("d.csv", raw, "text/csv")})
    packed = client.post("/api/datasets", headers={**headers, "X-Upload-Encoding": "gzip"},
                         files={"file": ("d.csv", gzip.compress(raw), "text/csv")})

    assert plain.status_code == 200
    assert packed.status_code == 200
    assert packed.json()["rows"] == plain.json()["rows"]
    assert packed.json()["column_names"] == plain.json()["column_names"]


def test_upload_without_the_header_is_still_accepted():
    """An older or simpler client that cannot compress must keep working."""
    client, headers = _client_with_session()
    response = client.post("/api/datasets", headers=headers,
                           files={"file": ("d.csv", _csv_bytes(), "text/csv")})
    assert response.status_code == 200


def test_a_decompression_bomb_is_refused():
    """The size limit applies to the expanded bytes, not the compressed ones.

    A few kilobytes of zeros expand into hundreds of megabytes. Checking only the
    compressed length would let that through and exhaust the process.
    """
    import gzip

    from backend.api import datasets

    bomb = gzip.compress(b"0" * (datasets.MAX_UPLOAD_BYTES + 1024))
    assert len(bomb) < 1024 * 1024, "the point is that it is small on the wire"

    client, headers = _client_with_session()
    response = client.post("/api/datasets", headers={**headers, "X-Upload-Encoding": "gzip"},
                           files={"file": ("d.csv", bomb, "text/csv")})

    assert response.status_code == 413


def test_a_corrupt_gzip_body_is_a_clean_error():
    """Garbage marked as gzip returns 422, not a 500."""
    client, headers = _client_with_session()
    response = client.post("/api/datasets", headers={**headers, "X-Upload-Encoding": "gzip"},
                           files={"file": ("d.csv", b"not gzip at all", "text/csv")})

    assert response.status_code == 422
    assert "Could not read the file" in response.json()["detail"]


# -- 2. Sessions must survive the process that created them -------------------

def test_session_survives_a_restart(store, dirty_df):
    """A new store rehydrates a checkpointed session, so the pipeline is not lost."""
    session = store.create()
    session.set("dataset", dirty_df)
    session.set("dataset_name", "air-quality.csv")
    session.set("target_column", "AQI")
    session.checkpoint()

    restarted = SessionStore()  # what the next process starts with
    assert restarted._sessions == {}

    revived = restarted.get(session.id)
    assert revived is not None, "a restart must not strand the browser's session id"
    assert revived.get("dataset_name") == "air-quality.csv"
    assert revived.get("dataset").shape == dirty_df.shape
    assert revived.completed_steps()["upload"] is True


def test_unknown_session_is_still_a_miss(store):
    """Rehydration must not invent sessions that were never created."""
    assert store.get("0" * 32) is None
    assert store.get(None) is None


def test_session_ids_cannot_escape_the_checkpoint_directory(store):
    """Ids are used to build paths, so anything but a uuid4 hex is refused."""
    assert store.get("../../../etc/passwd") is None
    assert persistence.load("../../../etc/passwd") is None
    assert persistence.save("../../evil", {"a": 1}) is False


def test_unreadable_checkpoint_is_discarded(store):
    """A checkpoint that cannot be loaded yields a clean miss, not an exception."""
    session = store.create()
    session.set("dataset_name", "x.csv")
    session.checkpoint()

    path = persistence.SESSION_DIR / f"{session.id}.pkl"
    assert path.exists()
    path.write_bytes(b"not a pickle")

    assert SessionStore().get(session.id) is None
    assert not path.exists(), "the unusable file should be cleaned up"


def test_checkpoint_failure_does_not_break_the_request(store, monkeypatch, tmp_path):
    """Persistence is a cache: if it cannot be written, the pipeline still works.

    The directory is pointed at an existing *file*, so `mkdir` fails the way a full or
    read-only disk would.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setattr(persistence, "SESSION_DIR", blocker)

    session = store.create()
    session.set("dataset_name", "x.csv")
    session.checkpoint()  # must not raise

    assert session.get("dataset_name") == "x.csv"


# -- 3. Dirty numeric columns must not become huge categories -----------------

def test_numeric_columns_with_stray_tokens_are_recovered(dirty_df):
    """A '-' in a readings column must not turn it into hundreds of categories."""
    coerced = _coerce_numeric_like(dirty_df)

    assert "PM10" in coerced
    assert pd.api.types.is_numeric_dtype(dirty_df["PM10"])
    assert dirty_df["PM10"].isna().sum() > 0, "unparseable entries become missing values"


def test_genuine_text_columns_are_left_alone(dirty_df):
    """Only columns that are overwhelmingly numeric are converted."""
    _coerce_numeric_like(dirty_df)
    assert not pd.api.types.is_numeric_dtype(dirty_df["City"])


def test_high_cardinality_text_is_not_offered_for_encoding(dirty_df):
    """Timestamps and identifiers are reported apart from real categories."""
    _coerce_numeric_like(dirty_df)
    profile = DatasetProfiler(
        dirty_df, target_column="AQI", problem_type="Regression"
    ).compute_profile()

    assert "From Date" in profile["high_cardinality_columns"]
    assert "From Date" not in profile["categorical_columns"]
    assert "City" in profile["categorical_columns"]
    assert profile["cardinality"]["From Date"] > HIGH_CARDINALITY_LIMIT


def test_advisor_drops_high_cardinality_columns(dirty_df):
    """The default action for an identifier column is drop, never encode.

    The category matters as much as the action. Only the imputer control offers "Drop
    Column", and the pipeline reads dropped columns from `impute_config` alone, so a drop
    filed under "encoding" is shown to the user and then silently ignored.
    """
    _coerce_numeric_like(dirty_df)
    profile = DatasetProfiler(
        dirty_df, target_column="AQI", problem_type="Regression"
    ).compute_profile()

    recs = [
        rec
        for rec in PreprocessingAdvisor().recommend(profile)
        if rec.metadata.get("column") == "From Date"
    ]
    actions = [rec.metadata.get("action") for rec in recs]

    assert "drop" in actions
    assert "ordinal" not in actions and "onehot" not in actions
    assert [rec.category for rec in recs if rec.metadata.get("action") == "drop"] == ["imputation"]


def test_a_dropped_column_is_not_also_scaled(dirty_df):
    """A column recommended for removal must not also carry a scaling recommendation.

    CO2 is 100% missing, so it is dropped - yet the UI showed "Standard Scale: CO2" and
    "Drop Column: CO2" in the same panel, which read as the tool contradicting itself.
    """
    _coerce_numeric_like(dirty_df)
    profile = DatasetProfiler(
        dirty_df, target_column="AQI", problem_type="Regression"
    ).compute_profile()

    co2 = [
        rec.metadata.get("action")
        for rec in PreprocessingAdvisor().recommend(profile)
        if rec.metadata.get("column") == "CO2"
    ]
    assert co2 == ["drop"], f"CO2 should only be dropped, got {co2}"


# -- 4. Reasoning models put their answer outside `content` -------------------

class _Response:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.parametrize("field", ["content", "reasoning_content", "reasoning"])
def test_reply_is_read_from_whichever_field_carries_it(field):
    """A model answering outside `content` is an answer, not an empty response."""
    message = {"content": "", field: "The narration."}
    client = LLMClient(api_key="k")

    assert client._parse(_Response({"choices": [{"message": message}]})) == "The narration."


def test_a_genuinely_empty_reply_is_still_reported():
    """The empty-response error must survive for the case it was written for."""
    client = LLMClient(api_key="k")
    payload = {"choices": [{"message": {"content": "", "reasoning": "  "}}]}

    assert client._parse(_Response(payload)) is None
    assert "empty response" in client._last_error
