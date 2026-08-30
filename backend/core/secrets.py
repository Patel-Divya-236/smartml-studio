"""Load LLM credentials from a secrets file.

`src/llm` reads its configuration from environment variables only, so this is the single
place that turns a file on disk into those variables.

`.streamlit/secrets.toml` remains in the candidate list purely for continuity: that is
where existing installs already keep their key, and silently ignoring it would look like
the key had stopped working. New installs should use `secrets.toml` at the repository
root.
"""

import logging
import os
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

# Checked in order; the first file that parses wins.
CANDIDATE_PATHS = (
    Path("secrets.toml"),
    Path(".secrets.toml"),
    Path(".streamlit/secrets.toml"),   # legacy location, still honoured
)


def load_llm_env(root: Path | None = None) -> list[str]:
    """Copy LLM settings from a secrets file into `os.environ`.

    Existing environment variables win, so an explicitly exported value is never
    overwritten. A missing or malformed file is not an error — the client reports itself
    unavailable and every caller falls back to its static text.

    Returns:
        The names of the keys this call actually set, for logging.
    """
    base = root or Path.cwd()
    loaded: list[str] = []

    for candidate in CANDIDATE_PATHS:
        path = base / candidate
        if not path.is_file():
            continue

        try:
            with path.open("rb") as handle:
                values = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        for key in KEYS:
            if os.environ.get(key):
                continue
            value = values.get(key)
            if value:
                os.environ[key] = str(value)
                loaded.append(key)

        # Never log the values themselves — one of them is an API key.
        logger.info("Loaded %d LLM setting(s) from %s.", len(loaded), path)
        return loaded

    logger.info("No secrets file found; LLM narration will report itself unavailable.")
    return loaded
