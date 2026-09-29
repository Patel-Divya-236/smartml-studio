"""Resident memory checks, so training refuses a job instead of being killed mid-run.

The deploy target is a 512MB instance. Going over it is not a slowdown: the host kills
the process. Because sessions live in a module-level dict, that kill takes every uploaded
dataset with it -- so a user who asked for one model too many comes back to "Complete the
upload step first" and re-uploads a file that was fine.

Checking first turns that into a refusal the session survives. `psutil` is the only way to
read RSS portably; when it is missing (it is a hard requirement, but a deploy can drift)
every check passes rather than blocking training on a monitoring dependency.
"""

import logging

from config.settings import SETTINGS

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard, exercised only by a broken install
    import psutil

    _PROCESS = psutil.Process()
except Exception:  # pragma: no cover
    _PROCESS = None
    logger.warning("psutil unavailable; training memory checks are disabled.")

# Reference frame for the per-model costs in SETTINGS. A model's cost is scaled against
# the rows it is actually given, since a boosting library's working set tracks the data.
REFERENCE_ROWS = 40_000

# Held back from the budget. The check happens before the job starts, and the request
# handling, the metrics frame and the progress events all allocate after it.
HEADROOM_MB = 40


def current_usage_mb() -> float:
    """Return this process's resident memory in MB, or 0.0 when it cannot be read."""
    if _PROCESS is None:
        return 0.0
    try:
        return _PROCESS.memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - a process that cannot read itself
        return 0.0


def estimate_cost_mb(models: list[str], rows: int) -> float:
    """Estimate what training these models on this many rows will add."""
    scale = max(rows, 1) / REFERENCE_ROWS
    return sum(
        SETTINGS.MODEL_MEMORY_COST_MB.get(name, SETTINGS.DEFAULT_MODEL_MEMORY_COST_MB)
        for name in models
    ) * scale


def models_that_fit(models: list[str], rows: int) -> list[str]:
    """Return the leading models that fit the budget, in the order given.

    Order is preserved rather than sorted cheapest-first: the caller asked for these, and
    silently reordering them would make the comparison table's contents depend on memory
    pressure. An empty list means not even the first one fits.
    """
    if _PROCESS is None:
        return list(models)

    available = SETTINGS.MEMORY_BUDGET_MB - current_usage_mb() - HEADROOM_MB
    scale = max(rows, 1) / REFERENCE_ROWS

    fitting: list[str] = []
    for name in models:
        cost = SETTINGS.MODEL_MEMORY_COST_MB.get(
            name, SETTINGS.DEFAULT_MODEL_MEMORY_COST_MB
        ) * scale
        if cost > available:
            break
        available -= cost
        fitting.append(name)
    return fitting
