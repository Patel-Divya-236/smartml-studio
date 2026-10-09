"""The dataset profile, condensed into what the planner needs to decide.

The full profile carries a complete correlation matrix and per-column dictionaries keyed
in four different places; sent as-is it is mostly noise and, on a wide table, thousands of
tokens of it. This digest is one entry per column with the statistics a preprocessing
decision actually turns on, plus the dataset-level facts a model choice turns on.

Only aggregates and column names leave the machine -- never a row, never a cell value.
That matches the narration payloads in `src/llm/context.py`.
"""

import math
from typing import Any

from config.settings import SETTINGS
from src.llm.context import strong_correlations

# Correlated pairs described to the planner. More than the narration payloads send,
# because pairs are what PCA and feature selection decisions are made from.
MAX_CORRELATED_PAIRS = 10
MAX_TARGET_CORRELATIONS = 8


def _number(value: Any, digits: int = 3) -> float | None:
    """A rounded float, or None for anything that is not a finite number.

    Skewness of a constant column is NaN, and NaN is not valid JSON -- `json.dumps` would
    write it anyway and the provider would reject the request.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def column_kind(column: str, profile: dict[str, Any]) -> str:
    """Classify a column the way the validator will judge it."""
    if column in set(profile.get("numeric_columns") or []):
        return "numeric"
    if column in set(profile.get("high_cardinality_columns") or []):
        return "high_cardinality_text"
    return "categorical"


def described_columns(profile: dict[str, Any], target: str | None) -> list[str]:
    """The non-target columns the planner is told about, in dataset order.

    Capped at SETTINGS.PLAN_MAX_COLUMNS. Columns past the cap are not lost: the rule-based
    advisor still recommends for them, and the merge fills them in.
    """
    columns = [c for c in (profile.get("dtypes") or {}) if c != target]
    return columns[: SETTINGS.PLAN_MAX_COLUMNS]


def build_planning_profile(profile: dict[str, Any], target: str | None) -> dict[str, Any]:
    """Return a compact, JSON-safe description of the dataset for the planner."""
    rows, total_columns = profile.get("shape", (0, 0))
    rows = int(rows or 0)
    missing_pct = profile.get("missing_pct") or {}
    cardinality = profile.get("cardinality") or {}
    skewness = profile.get("skewness") or {}
    numeric_min = profile.get("numeric_min") or {}
    outliers = profile.get("outliers") or {}
    dtypes = profile.get("dtypes") or {}

    all_features = [c for c in dtypes if c != target]
    described = described_columns(profile, target)

    columns: list[dict[str, Any]] = []
    for column in described:
        kind = column_kind(column, profile)
        entry: dict[str, Any] = {
            "name": str(column),
            "kind": kind,
            "missing_pct": _number(missing_pct.get(column, 0.0), 2) or 0.0,
            "distinct": int(cardinality.get(column, 0) or 0),
        }
        if kind == "numeric":
            outlier_count = int((outliers.get(column) or {}).get("count", 0) or 0)
            entry["skew"] = _number(skewness.get(column))
            entry["min"] = _number(numeric_min.get(column), 4)
            entry["outliers"] = outlier_count
            entry["outlier_pct"] = _number(outlier_count / rows * 100, 2) if rows else 0.0
        columns.append(entry)

    kinds = [column_kind(c, profile) for c in all_features]
    dataset: dict[str, Any] = {
        "rows": rows,
        "columns": int(total_columns or 0),
        "task": profile.get("problem_type"),
        "target": None if target is None else str(target),
        "duplicate_rows": int(profile.get("duplicates", 0) or 0),
        "numeric_features": kinds.count("numeric"),
        "categorical_features": kinds.count("categorical"),
        "high_cardinality_text_features": kinds.count("high_cardinality_text"),
    }
    if len(all_features) > len(described):
        dataset["columns_not_described"] = len(all_features) - len(described)

    digest: dict[str, Any] = {"dataset": dataset, "columns": columns}

    class_balance = profile.get("class_balance") or {}
    if class_balance:
        counts = {str(k): int(v) for k, v in class_balance.items()}
        total = sum(counts.values()) or 1
        smallest = min(counts.values()) if counts else 0
        digest["target_classes"] = {
            "count": len(counts),
            "share_pct": {k: _number(v / total * 100, 1) for k, v in counts.items()},
            # How many times larger the biggest class is than the smallest. Above
            # SETTINGS.CLASS_IMBALANCE_RATIO the data counts as imbalanced.
            "imbalance_ratio": _number(max(counts.values()) / smallest, 2) if smallest else None,
        }
    elif profile.get("target_summary"):
        digest["target_summary"] = {
            k: _number(v) for k, v in (profile.get("target_summary") or {}).items()
        }

    matrix = profile.get("correlation_matrix") or {}
    feature_matrix = {
        a: {b: r for b, r in (row or {}).items() if b != target}
        for a, row in matrix.items()
        if a != target
    }
    pairs = strong_correlations(feature_matrix, limit=MAX_CORRELATED_PAIRS)
    if pairs:
        digest["strongly_correlated_features"] = [
            {"a": str(a), "b": str(b), "r": _number(r, 2)} for a, b, r in pairs
        ]

    if target is not None and target in matrix:
        related = [
            (str(col), _number(r, 2))
            for col, r in (matrix.get(target) or {}).items()
            if col != target and _number(r) is not None
        ]
        related.sort(key=lambda pair: abs(pair[1]), reverse=True)
        if related:
            digest["correlation_with_target"] = [
                {"feature": col, "r": r} for col, r in related[:MAX_TARGET_CORRELATIONS]
            ]

    return digest
