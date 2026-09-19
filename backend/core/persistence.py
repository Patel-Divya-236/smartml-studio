"""Disk persistence for session state.

Sessions live in a module-level dict, so anything that ends the process — a redeploy, an
out-of-memory kill, a crash — used to take every uploaded dataset with it. The browser kept
its session id, the server no longer recognised it, and the user was told to "Complete the
upload step first" with no explanation. Checkpointing to disk closes that gap: the store
rehydrates a session on the first request after a restart and the pipeline carries on.

Pickle, not a schema, because `Session.data` holds live pandas frames and fitted sklearn
estimators. That makes the files a cache, not a database: they are only ever written and
read by this process on this machine, and every operation degrades to in-memory-only
behaviour on failure rather than failing the request.

Note the deployment implication: this survives a process restart only if the filesystem
outlives the process. On a host with an ephemeral container filesystem a spin-down still
discards everything; a persistent volume is what makes it durable.
"""

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSION_DIR = Path(os.environ.get("SMARTML_SESSION_DIR", ".sessions"))

# Pickle loads arbitrary objects, so only ids this process could have minted are accepted.
# A session id is a uuid4 hex; anything else must not become a filesystem path.
_ID_LENGTH = 32


def _is_safe_id(session_id: str) -> bool:
    """Reject anything that is not a uuid4 hex string, so no id can escape SESSION_DIR."""
    return len(session_id) == _ID_LENGTH and all(c in "0123456789abcdef" for c in session_id)


def _path_for(session_id: str) -> Path:
    """Return the checkpoint path for a session id."""
    return SESSION_DIR / f"{session_id}.pkl"


def save(session_id: str, payload: dict[str, Any]) -> bool:
    """Write a session's stage data to disk. Returns whether it was written.

    Writes to a temporary file and renames it into place, so a process killed mid-write
    leaves the previous good checkpoint intact rather than a truncated one.
    """
    if not _is_safe_id(session_id):
        return False

    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=SESSION_DIR, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_name, _path_for(session_id))
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
    except Exception as exc:
        # Never fail a request because the cache could not be written.
        logger.warning("Could not checkpoint session %s: %s", session_id[:8], exc)
        return False

    return True


def load(session_id: str) -> dict[str, Any] | None:
    """Read a session's stage data back, or None when there is no usable checkpoint."""
    if not _is_safe_id(session_id):
        return None

    path = _path_for(session_id)
    if not path.exists():
        return None

    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception as exc:
        # A checkpoint written by different library versions may no longer unpickle.
        # Discard it: a clean "upload again" beats a half-restored pipeline.
        logger.warning("Discarding unreadable checkpoint for %s: %s", session_id[:8], exc)
        delete(session_id)
        return None

    if not isinstance(payload, dict):
        delete(session_id)
        return None

    logger.info("Rehydrated session %s from disk.", session_id[:8])
    return payload


def delete(session_id: str) -> None:
    """Remove a session's checkpoint, ignoring a missing file."""
    if not _is_safe_id(session_id):
        return
    try:
        _path_for(session_id).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not delete checkpoint for %s: %s", session_id[:8], exc)
