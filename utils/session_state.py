"""Session-state management helpers for SmartML Studio.

Provides ``init_session_state()`` to bootstrap all shared keys with
safe defaults, plus typed ``get_state`` / ``set_state`` accessors
that include debug logging.
"""

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ── Canonical session-state keys and their defaults ────────────────
_SESSION_DEFAULTS: dict[str, Any] = {
    # Module 2 — Dataset Upload
    "dataset": None,                    # pd.DataFrame
    "dataset_name": None,               # str — original filename
    "target_column": None,              # str
    "problem_type": None,               # "Classification" | "Regression" | "Time Series"

    # Module 3 — Dataset Analysis
    "profile": None,                    # dict produced by DatasetProfiler

    # Module 4 — Visualization Advisor
    "viz_recommendations": None,        # list[Recommendation]
    "viz_selected": None,               # list[Recommendation] (user-accepted)

    # Module 5 — Preprocessing Advisor
    "preprocessing_recommendations": None,  # list[Recommendation]
    "preprocessed_data": None,              # pd.DataFrame

    # Module 6 — Feature Engineering
    "feature_engineered_data": None,    # pd.DataFrame

    # Module 7 — Model Advisor
    "model_recommendations": None,      # list[Recommendation]

    # Module 8 — Model Training
    "trained_models": None,             # dict[str, trained model info]
    "X_train": None,                    # Training features
    "X_test": None,                     # Test features
    "y_train": None,                    # Training labels
    "y_test": None,                     # Test labels

    # Module 9 — Model Comparison
    "model_comparison": None,           # dict / DataFrame of comparison metrics

    # Module 10 — Prediction
    "predictions": None,                # pd.DataFrame
    "ensemble": None,                   # HybridEnsemble instance
    "selected_prediction_model": None,  # str — model name or "ensemble"

    # Module 11 — Explainability
    "explainability_results": None,     # dict of SHAP outputs

    # Module 12 — Download / Report
    "evaluation_report": None,          # dict bundling all module outputs
}


def init_session_state() -> None:
    """Initialise all session-state keys with safe defaults.

    Idempotent — only sets keys that do not already exist so that
    Streamlit hot-reloads never clobber user progress.
    """
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    logger.debug("Session state initialised (%d keys).", len(_SESSION_DEFAULTS))


def get_state(key: str) -> Any:
    """Return a session-state value by key.

    Args:
        key: One of the canonical keys defined in ``_SESSION_DEFAULTS``.

    Returns:
        The current value, or ``None`` if the key was never set.
    """
    value = st.session_state.get(key)
    logger.debug("get_state(%s) -> %s", key, type(value).__name__)
    return value


def set_state(key: str, value: Any) -> None:
    """Set a session-state value by key with logging.

    Args:
        key: One of the canonical keys defined in ``_SESSION_DEFAULTS``.
        value: The value to store.
    """
    st.session_state[key] = value
    logger.debug("set_state(%s) <- %s", key, type(value).__name__)


def reset_downstream(from_key: str) -> None:
    """Reset all session-state keys that come *after* ``from_key``.

    Useful when a user re-uploads data or changes the target column —
    everything downstream must be invalidated to prevent stale state.

    Args:
        from_key: The key whose downstream dependants should be cleared.
    """
    keys = list(_SESSION_DEFAULTS.keys())
    if from_key not in keys:
        logger.warning("reset_downstream: unknown key '%s'.", from_key)
        return

    start_idx = keys.index(from_key) + 1
    for key in keys[start_idx:]:
        st.session_state[key] = _SESSION_DEFAULTS[key]
    logger.info("Session state reset from '%s' onward (%d keys).", from_key, len(keys) - start_idx)
