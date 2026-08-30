"""Shared request dependencies."""

from fastapi import Header, HTTPException

from backend.core.session import STORE, Session


def get_session(x_session_id: str | None = Header(default=None)) -> Session:
    """Resolve the caller's session from the `X-Session-Id` header.

    A missing or expired id is a 400 rather than a silent new session: quietly handing
    back a blank session would make a stale browser tab look like it had simply lost its
    dataset, with no indication why.
    """
    session = STORE.get(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /api/session first.",
        )
    return session


def require(session: Session, key: str, step: str) -> object:
    """Return a stage value, or explain which step must be completed first."""
    value = session.get(key)
    if value is None:
        raise HTTPException(status_code=409, detail=f"Complete the {step} step first.")
    return value
