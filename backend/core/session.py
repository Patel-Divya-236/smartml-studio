"""Server-side session store for the React front end.

This replaces `st.session_state`. The key ordering below is load-bearing and is copied
deliberately from `utils/session_state.py`: `reset_downstream` invalidates every stage
*after* the one that changed, which is what stops a user editing preprocessing and then
reading metrics computed from the previous split.

Storage is in-memory and process-local, backed by a disk checkpoint (`backend.core.
persistence`) so a restart does not silently discard a user's uploaded dataset and send
them back to the upload step. Sessions expire after `SESSION_TTL_SECONDS` of inactivity so
a long-running server does not accumulate DataFrames indefinitely. There is still no
database because there is no login and nothing needs to outlive the browser tab; the plan
file records the trade-off if experiment tracking is wanted later.
"""

import logging
import threading
import time
import uuid
from typing import Any

from backend.core import persistence

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 60 * 60 * 4  # four hours of inactivity

# Ordered stage keys. Order defines the invalidation cascade — do not reorder casually.
STAGE_KEYS: list[str] = [
    # Dataset
    "dataset", "dataset_name", "target_column", "problem_type",
    # Analysis
    "profile",
    # Visualization
    "viz_recommendations", "viz_selected",
    # Preprocessing
    "preprocessing_recommendations", "test_size", "label_encoder",
    "preprocessing_config", "preprocessed_train", "preprocessed_test",
    "y_train", "y_test", "preprocessing_pipeline",
    # Feature engineering
    "feature_config", "feature_engineered_train", "feature_engineered_test",
    "feature_names",
    # Model advisor
    "model_recommendations", "selected_models",
    # Training
    "X_train", "X_test", "trained_models", "training_failures",
    # Comparison
    "model_comparison",
    # Prediction
    "predictions", "ensemble", "selected_prediction_model",
    # Explainability
    "explainability_results",
    # Report
    "evaluation_report",
]

# Which pipeline step each stage key belongs to, for the UI's step-completion rail.
STEP_REQUIREMENTS: dict[str, list[str]] = {
    "upload": ["dataset"],
    "analysis": ["profile"],
    "visualization": ["viz_recommendations"],
    "preprocessing": ["preprocessed_train"],
    "features": ["feature_engineered_train"],
    "model-advisor": ["selected_models"],
    "training": ["trained_models"],
    "comparison": ["model_comparison"],
    "prediction": ["predictions"],
    "explainability": ["explainability_results"],
    "download": ["trained_models"],
}


class Session:
    """One user's pipeline state."""

    def __init__(self, session_id: str) -> None:
        """Create an empty session with every stage key set to None."""
        self.id = session_id
        self.created_at = time.time()
        self.last_seen = time.time()
        self.data: dict[str, Any] = {key: None for key in STAGE_KEYS}

    def get(self, key: str) -> Any:
        """Return a stage value, or None when never set."""
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a stage value."""
        self.data[key] = value

    def reset_downstream(self, from_key: str) -> None:
        """Clear every stage that comes after `from_key`.

        Called whenever an upstream choice changes, so the UI can grey out the steps
        whose results no longer correspond to the current configuration.
        """
        if from_key not in STAGE_KEYS:
            logger.warning("reset_downstream: unknown key %r.", from_key)
            return

        start = STAGE_KEYS.index(from_key) + 1
        for key in STAGE_KEYS[start:]:
            self.data[key] = None
        logger.info("Session %s reset from %r onward.", self.id[:8], from_key)

    def completed_steps(self) -> dict[str, bool]:
        """Report which pipeline steps have produced their output."""
        return {
            step: all(self.data.get(k) is not None for k in keys)
            for step, keys in STEP_REQUIREMENTS.items()
        }

    def checkpoint(self) -> None:
        """Persist this session so it survives a process restart.

        Call it after the writes that complete a stage, not after every `set`. Failure is
        logged and swallowed inside `persistence.save` — a session that cannot be written
        still works for as long as the process lives.
        """
        persistence.save(self.id, self.data)


class SessionStore:
    """Thread-safe registry of active sessions."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        """Create and register a new session."""
        session = Session(uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        logger.info("Session %s created.", session.id[:8])
        return session

    def get(self, session_id: str | None) -> Session | None:
        """Return a live session, refreshing its activity timestamp.

        A miss falls back to the disk checkpoint before giving up. That fallback is the
        whole reason a restart no longer strands the user on "Complete the upload step
        first" holding a session id the server has forgotten.
        """
        if not session_id:
            return None

        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = time.time()
                return session

        restored = self._rehydrate(session_id)
        if restored is None:
            logger.info("Session %s not found, and no checkpoint on disk.", session_id[:8])
        return restored

    def _rehydrate(self, session_id: str) -> Session | None:
        """Rebuild a session from its checkpoint, or return None when there is none."""
        payload = persistence.load(session_id)
        if payload is None:
            return None

        session = Session(session_id)
        # Assign key by key so a checkpoint written before a stage key was added or removed
        # still loads, instead of resurrecting a stale shape.
        for key in STAGE_KEYS:
            session.data[key] = payload.get(key)

        with self._lock:
            # Another request may have rehydrated the same id while this one was reading.
            existing = self._sessions.get(session_id)
            if existing is not None:
                existing.last_seen = time.time()
                return existing
            self._sessions[session_id] = session

        return session

    def get_or_create(self, session_id: str | None) -> Session:
        """Return the named session, or a fresh one when it is missing or expired."""
        return self.get(session_id) or self.create()

    def purge_expired(self) -> int:
        """Drop sessions idle for longer than the TTL. Returns how many were removed.

        Also removes checkpoints — both for expired in-memory sessions and for orphaned
        files left by a previous process, which nothing else would ever clean up.
        """
        cutoff = time.time() - SESSION_TTL_SECONDS
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in stale:
                del self._sessions[sid]
            live = set(self._sessions)

        for sid in stale:
            persistence.delete(sid)

        orphans = 0
        try:
            for path in persistence.SESSION_DIR.glob("*.pkl"):
                if path.stem in live or path.stem in stale:
                    continue
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    orphans += 1
        except Exception as exc:
            logger.warning("Could not sweep session checkpoints: %s", exc)

        if stale or orphans:
            logger.info("Purged %d expired session(s), %d orphaned file(s).", len(stale), orphans)
        return len(stale)


STORE = SessionStore()
