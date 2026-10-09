"""Application settings — config-driven thresholds and constants.

All tuneable thresholds live here so no magic numbers appear in
business logic. Import the module-level SETTINGS instance:

    from config.settings import SETTINGS
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppSettings:
    """Centralised, immutable application configuration.

    Every numeric threshold used by profilers, advisors, or UI logic
    is defined here rather than hard-coded in business logic.
    """

    # ── Training resources ─────────────────────────────────────────
    TRAINING_THREADS: int = int(os.environ.get("SMARTML_TRAINING_THREADS", "1"))
    """Threads each boosting library may use.

    One by default because the deploy target grants 0.1 of a CPU: a library that reads
    the host's core count and starts twelve workers only adds contention there. Measured
    per model in a fresh process on a 40,000-row frame, the memory difference between one
    thread and twelve is small -- XGBoost 114MB against 111MB, CatBoost 17MB against 23MB
    -- so this is a CPU setting, not the fix for an out-of-memory kill. MEMORY_BUDGET_MB
    below is that. Raise this on a host with real cores (see DEPLOYMENT.md).
    """

    MEMORY_BUDGET_MB: int = int(os.environ.get("SMARTML_MEMORY_BUDGET_MB", "512"))
    """Resident memory this process may reach before a training job is refused.

    Matches the free Render instance this deploys to. The point is not to save memory --
    it is to fail in a way that keeps the session. Measured on a 50,000-row, 13MB upload,
    the process reaches ~495MB by the end of training: 211MB of that is imports before any
    data arrives, ~39MB is the session itself, and the rest is what the boosting libraries
    allocate and never return (XGBoost alone accounts for ~114MB).

    Going over does not slow anything down; the host kills the process. That takes the
    in-memory session store with it, so every user sees "Complete the upload step first"
    and re-uploads. Refusing one job costs a model instead of the whole pipeline.

    Raise it to match the instance on a bigger host (see DEPLOYMENT.md).
    """

    MODEL_MEMORY_COST_MB: dict[str, int] = field(default_factory=lambda: {
        # Measured per model in its own process, fitting 40,000 rows x 29 features and
        # predicting, counting only what is still resident afterwards.
        "XGBoost": 114,
        "CatBoost": 23,
        "Random Forest": 8,
        "LightGBM": 5,
        "SVM": 5,
        "KNN": 12,   # keeps the training set; that is the algorithm, not overhead
    })
    """What each model costs, used to decide whether a job fits the budget."""

    DEFAULT_MODEL_MEMORY_COST_MB: int = 10
    """Assumed cost for a model not listed above -- the cheap sklearn estimators."""

    # ── Dataset Profiling ──────────────────────────────────────────
    HIGH_CARDINALITY_THRESHOLD: int = 20
    """Columns with more unique values than this are 'high-cardinality'."""

    MAX_UNIQUE_VALUES_FOR_CLASSIFICATION: int = 10
    """Target numeric column with unique values <= this is auto-detected as Classification."""

    SKEWNESS_THRESHOLD: float = 1.0
    """Absolute skewness above this triggers a skew warning."""

    OUTLIER_IQR_MULTIPLIER: float = 1.5
    """IQR multiplier for outlier detection (1.5 = standard, 3.0 = extreme)."""

    LOW_VARIANCE_THRESHOLD: float = 0.01
    """Features with variance below this are candidates for removal."""

    MISSING_VALUE_CONCERN_PCT: float = 5.0
    """Missing-value percentage above which the advisor flags concern."""

    CORRELATION_STRONG_THRESHOLD: float = 0.7
    """Absolute correlation above this is considered 'strong'."""

    CLASS_IMBALANCE_RATIO: float = 3.0
    """Majority-to-minority class ratio above this triggers imbalance warning."""

    # ── Visualization Advisor ──────────────────────────────────────
    MAX_CATEGORIES_FOR_PIE: int = 8
    """Pie charts are only recommended when unique values ≤ this."""

    MAX_CATEGORIES_FOR_BAR: int = 30
    """Bar charts are only recommended when unique values ≤ this."""

    # ── Preprocessing Advisor ─────────────────────────────────────
    MISSING_LOW_PCT: float = 5.0
    """Below this %, simple imputation is usually sufficient."""

    MISSING_HIGH_PCT: float = 30.0
    """Above this %, consider dropping the column."""

    # ── Model Advisor ──────────────────────────────────────────────
    SMALL_DATASET_ROWS: int = 1_000
    """Datasets with fewer rows than this are 'small'."""

    MEDIUM_DATASET_ROWS: int = 10_000
    """Datasets with fewer rows than this are 'medium'."""

    LARGE_DATASET_ROWS: int = 100_000
    """Datasets with more rows than this are 'large'."""

    # ── Model Training ─────────────────────────────────────────────
    DEFAULT_TEST_SIZE: float = 0.2
    """Default train/test split ratio."""

    DEFAULT_RANDOM_STATE: int = 42
    """Default random seed for reproducibility."""

    DEFAULT_CV_FOLDS: int = 5
    """Default number of cross-validation folds."""

    # ── UI / UX ────────────────────────────────────────────────────
    CONFIDENCE_STAR_THRESHOLDS: list[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    """Confidence-to-star-rating breakpoints (1★ … 5★)."""

    # ── LLM Explanation Layer ──────────────────────────────────────
    # Narration: the LLM explains decisions that have already been made. The planning
    # layer below is separate and is the only place it proposes them.
    # Any OpenAI-compatible provider works — set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
    # in the environment (or secrets.toml) to point at a different one.

    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    """OpenAI-compatible chat-completions base URL. Overridable via LLM_BASE_URL."""

    LLM_MODEL: str = "openai/gpt-oss-120b"
    """Model id on the chosen provider. Model ids change — if requests 404, list the
    provider's current models with `python scripts_list_models.py` and set LLM_MODEL.

    This is a reasoning model: it returns a `reasoning` field alongside `content`, and can
    leave `content` empty. LLMClient._parse falls back to that field rather than reporting
    the reply as empty. Verify any replacement against the provider's live catalogue —
    ids are retired without notice."""

    LLM_MAX_TOKENS: int = 2000
    """Output cap per narration.

    The prose itself is short, but a reasoning model spends part of this budget thinking
    before it writes. A hard prompt was measured using 882 of the previous 900-token cap,
    leaving the answer almost no room — and an answer squeezed out entirely is what reaches
    the user as an empty response."""

    LLM_TEMPERATURE: float = 0.3
    """Low temperature: narration should track the numbers, not embellish them."""

    LLM_TIMEOUT_SECONDS: float = 60.0
    """Hard timeout per request. On timeout the caller falls back to static text."""

    LLM_MAX_SHAP_FEATURES: int = 10
    """Top-N SHAP attributions (by absolute magnitude) sent per prediction."""

    LLM_MAX_CORRELATIONS: int = 5
    """Top-N strongest feature correlations included in the report payload."""

    # ── LLM Planning Layer ─────────────────────────────────────────
    # The planner is the one place an LLM *proposes* decisions. Everything it proposes
    # passes a deterministic validator (src/planning/validator.py) before a user sees it,
    # and the rule-based advisors fill whatever it skips or gets rejected.

    LLM_PLAN_MAX_TOKENS: int = 6000
    """Output cap for one planning call.

    Far above the narration cap because the reply is a JSON plan covering every column,
    and a reasoning model spends part of the same budget thinking first. A plan cut off
    mid-object is unparseable, so running out here costs the whole plan, not a sentence."""

    LLM_PLAN_TEMPERATURE: float = 0.1
    """Near-deterministic: the same profile should produce the same plan."""

    LLM_PLAN_TIMEOUT_SECONDS: float = 45.0
    """Hard timeout for the planning call. On timeout the rule-based advisors take over."""

    LLM_PLAN_REASONING_EFFORT: str = os.environ.get("LLM_PLAN_REASONING_EFFORT", "low")
    """Passed as `reasoning_effort` to reasoning models, which cuts planning latency.

    Providers that do not recognise the field reject the request with HTTP 400; the
    client then retries once without it, so an unsupported value costs a round trip,
    never the plan. Set it to an empty string to stop sending it."""

    PLAN_MAX_COLUMNS: int = 60
    """Columns described to the planner. Wider datasets send the first 60 non-target
    columns; the rule-based advisor covers the rest, so nothing is left unconfigured."""

    PLAN_MAX_REASON_CHARS: int = 240
    """Longest reason kept from the planner. Longer text is truncated, not rejected."""

    ONEHOT_MAX_CARDINALITY: int = 20
    """Above this many categories, one-hot encoding is rejected by the validator: every
    category becomes a column, and wide one-hot frames are what exhausted memory on the
    512MB deploy target."""

    POLY_MAX_NUMERIC: int = 10
    """Polynomial features are only accepted up to this many numeric columns. Degree 2 on
    n inputs makes (n+1)(n+2)/2 outputs, so 10 inputs is 66 columns and 30 would be 496."""

    SVM_MAX_ROWS: int = 20_000
    """Above this many rows, the validator rejects an SVM recommendation as unsafe to run
    by default. SVC is quadratic in rows: 50,000 rows took 374s on the deploy target,
    longer than every other model combined. Users can still tick it by hand."""

    MIN_RECOMMENDED_MODELS: int = 3
    """When fewer validated models than this survive, the rule ranking tops the list up."""


# ── Module-level singleton ─────────────────────────────────────────
SETTINGS = AppSettings()
