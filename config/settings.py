"""Application settings — config-driven thresholds and constants.

All tuneable thresholds live here so no magic numbers appear in
business logic. Import the module-level SETTINGS instance:

    from config.settings import SETTINGS
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppSettings:
    """Centralised, immutable application configuration.

    Every numeric threshold used by profilers, advisors, or UI logic
    is defined here rather than hard-coded in business logic.
    """

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


# ── Module-level singleton ─────────────────────────────────────────
SETTINGS = AppSettings()
