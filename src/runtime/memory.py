"""Resident memory checks, so training refuses a job instead of being killed mid-run.

The deploy target is a 512MB instance. Going over it is not a slowdown: the host kills
the process. Because sessions live in a module-level dict, that kill takes every uploaded
dataset with it -- so a user who asked for one model too many comes back to "Complete the
upload step first" and re-uploads a file that was fine.

Checking first turns that into a refusal the session survives. `psutil` is the only way to
read RSS portably; when it is missing (it is a hard requirement, but a deploy can drift)
every check passes rather than blocking training on a monitoring dependency.
"""

import ctypes
import gc
import logging
import sys

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

# Whether freed heap can actually be handed back, which decides whether model costs stack.
# glibc's malloc_trim is the mechanism; the deploy target is Linux, a Windows dev box is not.
RECLAIMS_MEMORY = sys.platform.startswith("linux")


def release_memory() -> float:
    """Return freed heap to the operating system. Returns the MB reclaimed.

    This is what lets a 512MB instance train the whole model set on 40,000 rows instead
    of two models at a time, and it works because of what the memory actually is.
    Measured on a 40,000-row frame, an XGBoost fit grows the process by 112MB while the
    fitted booster itself is 0.4MB, and deleting the model returns almost none of it. But
    a *second* XGBoost fit costs nothing -- the arena is reused. So the cost is per
    library, not per model, and the problem is only that CatBoost then allocates its own
    ~100MB beside XGBoost's rather than into it.

    glibc keeps that freed memory in the process heap because releasing and re-acquiring
    it from the kernel is usually wasteful. `malloc_trim` overrides that judgement, which
    is the right call here: the next model wants a different allocator's arena, not this
    one's, and the instance is measured against RSS.

    Linux and glibc only. Everywhere else this is the garbage collection alone, which is
    why the budget check in `models_that_fit` stays as the backstop rather than being
    replaced by this.
    """
    before = current_usage_mb()
    gc.collect()

    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as exc:  # pragma: no cover - musl, or a libc without it
            logger.debug("malloc_trim unavailable: %s", exc)

    reclaimed = before - current_usage_mb()
    if reclaimed > 1:
        logger.info("Released %.0fMB back to the OS.", reclaimed)
    return reclaimed


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


def _cost_of(name: str, scale: float) -> float:
    """What one model adds, scaled to the rows it will actually be given."""
    return SETTINGS.MODEL_MEMORY_COST_MB.get(
        name, SETTINGS.DEFAULT_MODEL_MEMORY_COST_MB
    ) * scale


def models_that_fit(models: list[str], rows: int) -> list[str]:
    """Return the leading models that fit the budget, in the order given.

    Where `release_memory` works, models do not accumulate: each library's arena is handed
    back before the next one allocates, so the peak is the baseline plus the single most
    expensive model rather than the sum of all of them. That is the difference between a
    40,000-row run training every model and being told to do it two at a time -- 112MB of
    headroom needed instead of 250MB.

    Where it does not work, the costs are summed, because they really will stack.

    Order is preserved rather than sorted cheapest-first: the caller asked for these, and
    silently reordering them would make the comparison table's contents depend on memory
    pressure. An empty list means not even the first one fits.
    """
    if _PROCESS is None:
        return list(models)

    available = SETTINGS.MEMORY_BUDGET_MB - current_usage_mb() - HEADROOM_MB
    scale = max(rows, 1) / REFERENCE_ROWS

    if RECLAIMS_MEMORY:
        # Only the largest arena is ever live at once, so admit every model up to the
        # first one that could not fit even on its own.
        fitting: list[str] = []
        for name in models:
            if _cost_of(name, scale) > available:
                break
            fitting.append(name)
        return fitting

    fitting = []
    for name in models:
        cost = _cost_of(name, scale)
        if cost > available:
            break
        available -= cost
        fitting.append(name)
    return fitting
