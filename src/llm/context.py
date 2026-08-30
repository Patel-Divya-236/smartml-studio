"""Payload builders — the privacy boundary for the LLM layer.

Every prompt sent to a provider is assembled here, and nowhere else. That makes one
rule enforceable in one file and testable in one test:

    Only aggregate statistics, metric tables, and — for a per-prediction explanation —
    the single already-preprocessed row the user explicitly asked about ever leave the
    machine. The uploaded dataset never does.

Keeping this in one module also bounds prompt size independently of dataset size, which
is what keeps the workload inside a provider's free tier.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


def summarise_profile(profile: dict[str, Any]) -> str:
    """Condense a DatasetProfiler profile into a few lines of aggregate statistics.

    Deliberately lossy: shape, missingness, dtype counts, target summary and the few
    strongest correlations. Never per-row values, never the full correlation matrix.
    """
    rows, cols = profile.get("shape", (0, 0))
    numeric = profile.get("numeric_columns", []) or []
    categorical = profile.get("categorical_columns", []) or []
    missing_pct = profile.get("missing_pct", {}) or {}
    skewness = profile.get("skewness", {}) or {}

    lines = [
        f"Rows: {rows}",
        f"Columns: {cols} ({len(numeric)} numeric, {len(categorical)} categorical)",
        f"Duplicate rows: {profile.get('duplicates', 0)}",
        f"Task type: {profile.get('problem_type', 'unknown')}",
    ]

    incomplete = {c: p for c, p in missing_pct.items() if p > 0}
    if incomplete:
        worst = sorted(incomplete.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append(
            "Columns with missing values: "
            + ", ".join(f"{c} ({p:.1f}%)" for c, p in worst)
        )
    else:
        lines.append("Columns with missing values: none")

    skewed = [c for c, s in skewness.items() if abs(s) > SETTINGS.SKEWNESS_THRESHOLD]
    if skewed:
        lines.append(f"Strongly skewed features: {', '.join(skewed[:5])}")

    class_balance = profile.get("class_balance", {}) or {}
    if class_balance:
        total = sum(class_balance.values()) or 1
        parts = [f"{k}: {v} ({v / total * 100:.1f}%)" for k, v in list(class_balance.items())[:8]]
        lines.append("Target class balance — " + "; ".join(parts))

    target_summary = profile.get("target_summary", {}) or {}
    if target_summary and not class_balance:
        lines.append(
            "Target distribution — "
            + ", ".join(f"{k}: {v:.4g}" for k, v in target_summary.items())
        )

    strong = _strong_correlations(profile.get("correlation_matrix", {}) or {})
    if strong:
        lines.append(
            "Strongest feature correlations — "
            + "; ".join(f"{a} vs {b}: r={r:.2f}" for a, b, r in strong)
        )

    return "\n".join(lines)


def build_prediction_payload(
    *,
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    predicted_label: Any,
    actual_label: Any = None,
    problem_type: str = "Classification",
    base_value: float | None = None,
    output_space: str | None = None,
) -> str:
    """Format one prediction's SHAP attributions as a compact table.

    Only the single row being explained is included, and its values are the
    already-preprocessed ones the model actually saw — not the original upload.

    Args:
        feature_names: Model input column names.
        feature_values: Transformed feature values for this one sample.
        shap_values: SHAP contribution per feature for this one sample.
        predicted_label: What the model predicted.
        actual_label: Ground truth, when known.
        problem_type: "Classification" or "Regression".
        base_value: The model's expected value before feature contributions.
        output_space: Units the contributions are in, from
            ``explainability.explainer.infer_output_space``. Stated explicitly so the
            model cannot silently reinterpret log-odds as percentage points.

    Returns:
        A text block listing the top-N features by absolute contribution.
    """
    values = np.asarray(feature_values, dtype=float).ravel()
    contributions = np.asarray(shap_values, dtype=float).ravel()

    n = min(len(feature_names), len(values), len(contributions))
    if n == 0:
        logger.warning("build_prediction_payload received no usable features.")
        return "No feature attributions were available for this prediction."

    order = np.argsort(np.abs(contributions[:n]))[::-1][: SETTINGS.LLM_MAX_SHAP_FEATURES]

    lines = [f"Task: {problem_type}", f"Model prediction: {predicted_label}"]
    if actual_label is not None:
        lines.append(f"Actual value: {actual_label}")
    baseline = _as_scalar(base_value)
    if baseline is not None:
        lines.append(f"Model baseline (expected value before features): {baseline:.4f}")

    lines.append("")
    lines.append("Feature contributions (SHAP), largest absolute effect first.")
    lines.append("A positive contribution pushes the prediction up, negative pushes it down.")
    lines.append(f"Contribution units: {output_space or 'unspecified model output units'}")
    lines.append("")
    lines.append(
        "The 'vs largest' column is each contribution's size relative to the biggest "
        "one, precomputed. Use it directly rather than doing your own arithmetic."
    )
    lines.append("")
    lines.append(
        f"{'feature':<32} {'value':>14} {'contribution':>14} {'vs largest':>11}"
    )

    largest = float(np.max(np.abs(contributions[:n]))) or 1.0
    for idx in order:
        share = abs(contributions[idx]) / largest
        lines.append(
            f"{str(feature_names[idx])[:32]:<32} {values[idx]:>14.4f} "
            f"{contributions[idx]:>+14.4f} {share:>10.2f}x"
        )

    omitted = n - len(order)
    if omitted > 0:
        lines.append(f"\n({omitted} further features had smaller contributions and are omitted.)")

    return "\n".join(lines)


def build_report_payload(
    *,
    dataset_name: str,
    target_column: str,
    problem_type: str,
    profile: dict[str, Any],
    comparison: pd.DataFrame | None,
    report_state: dict[str, Any] | None = None,
) -> str:
    """Assemble the dataset profile and model comparison for report narration.

    Aggregates and the metrics table only — no dataset rows at any point.
    """
    sections = [
        "DATASET",
        f"File: {dataset_name}",
        f"Target column: {target_column}",
        "",
        summarise_profile(profile),
        "",
        "MODEL COMPARISON",
    ]

    if comparison is not None and not comparison.empty:
        sections.append(comparison.round(4).to_string(index=False))
    else:
        sections.append("No models were evaluated.")

    state = report_state or {}
    if state.get("prediction_mode"):
        sections += [
            "",
            "FINAL SELECTION",
            f"Strategy: {state.get('selected_model', 'not recorded')}",
        ]
        if state.get("ensemble_voting"):
            members = ", ".join(state.get("ensemble_models", []) or []) or "none"
            sections.append(f"Ensemble voting: {state['ensemble_voting']} over {members}")

    return "\n".join(sections)



def build_global_payload(
    *,
    model_name: str,
    ranked_importance: list[tuple[str, float]],
    problem_type: str,
    n_samples: int,
    output_space: str | None = None,
) -> str:
    """Format the global feature-importance ranking behind the SHAP summary plot.

    ``ranked_importance`` is mean absolute SHAP per feature -- how much each feature
    moves this model's output on average, ignoring direction. That is the quantity the
    beeswarm plot encodes as horizontal spread, which is the part non-specialists
    reliably misread.
    """
    if not ranked_importance:
        return "No global feature importances were available for this model."

    total = sum(v for _, v in ranked_importance) or 1.0
    lines = [
        f"Model: {model_name}",
        f"Task: {problem_type}",
        f"Explained samples: {n_samples}",
        f"Contribution units: {output_space or 'unspecified model output units'}",
        "",
        "Mean absolute SHAP value per feature (average influence on the model's output,",
        "direction ignored). Share is that feature's portion of the listed total.",
        "",
        f"{'feature':<32} {'mean |SHAP|':>14} {'share':>8}",
    ]
    for name, value in ranked_importance:
        lines.append(f"{name[:32]:<32} {value:>14.4f} {value / total * 100:>7.1f}%")
    return "\n".join(lines)


def build_recommendation_payload(
    *,
    label: str,
    category: str,
    reason: str,
    why_explanation: str,
    confidence_score: float,
    metadata: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> str:
    """Format one advisor recommendation for beginner-facing expansion.

    The recommendation has already been decided by the rule-based advisor. Everything
    here is that decision plus the dataset facts it was derived from -- the model is
    asked to explain it, never to revisit it.
    """
    lines = [
        "ADVISOR RECOMMENDATION (already decided by a rule -- explain it, do not revise it)",
        f"Recommendation: {label}",
        f"Category: {category or 'general'}",
        f"Confidence: {confidence_score:.2f} of 1.00",
        f"Rule's stated reason: {reason}",
        f"Rule's stated rationale: {why_explanation}",
    ]

    meta = metadata or {}
    column = meta.get("column")
    if meta:
        rendered = ", ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        if rendered:
            lines.append(f"Rule parameters: {rendered}")

    if profile and column:
        lines.append("")
        lines.append(f"COLUMN FACTS FOR '{column}'")
        for key, source in (
            ("dtype", "dtypes"),
            ("missing %", "missing_pct"),
            ("distinct values", "cardinality"),
            ("skewness", "skewness"),
        ):
            table = profile.get(source) or {}
            if column in table:
                lines.append(f"- {key}: {_format_stat(table[column])}")
        outliers = (profile.get("outliers") or {}).get(column)
        if isinstance(outliers, dict) and "count" in outliers:
            lines.append(f"- outliers flagged: {outliers['count']}")

    if profile:
        rows, cols = profile.get("shape", (0, 0))
        lines.append("")
        lines.append(f"DATASET CONTEXT: {rows} rows, {cols} columns, "
                     f"task type {profile.get('problem_type', 'unknown')}")

    return "\n".join(lines)



def build_column_payload(
    *,
    column: str,
    recommendations: list[Any],
    profile: dict[str, Any] | None = None,
) -> str:
    """Format every advisor recommendation that applies to one column.

    A single column often attracts several steps -- impute, then encode or scale. They
    are explained together because that is how they are applied, and because the reason
    for one frequently depends on another.
    """
    lines = [
        f"COLUMN: {column}",
        "The following preparation steps were chosen for this column by a rule-based",
        "advisor. Explain them; do not revise them.",
        "",
    ]

    for rec in recommendations:
        meta = getattr(rec, "metadata", {}) or {}
        action = meta.get("action", "")
        lines.append(f"- {rec.label}")
        lines.append(f"    step type: {rec.category or 'general'}"
                     + (f", action: {action}" if action else ""))
        lines.append(f"    confidence: {rec.confidence_score:.2f}")
        lines.append(f"    rule's reason: {rec.reason}")

    if profile:
        lines.append("")
        lines.append("COLUMN STATISTICS")
        for key, source in (
            ("dtype", "dtypes"),
            ("missing %", "missing_pct"),
            ("distinct values", "cardinality"),
            ("skewness", "skewness"),
        ):
            table = profile.get(source) or {}
            if column in table:
                lines.append(f"- {key}: {_format_stat(table[column])}")
        outliers = (profile.get("outliers") or {}).get(column)
        if isinstance(outliers, dict) and "count" in outliers:
            lines.append(f"- outliers flagged: {outliers['count']}")

        rows, cols = profile.get("shape", (0, 0))
        lines.append(f"- dataset size: {rows} rows, {cols} columns")

    return "\n".join(lines)

def _format_stat(value: Any) -> str:
    """Render a profile statistic without implying false precision."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)

def _as_scalar(value: Any) -> float | None:
    """Coerce a SHAP base value to a float, or None if it cannot be represented.

    Multiclass explainers return one expected value per class, so this takes the
    first entry rather than raising on an array.
    """
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return float(arr[0])


def _strong_correlations(
    correlation_matrix: dict[str, dict[str, float]],
) -> list[tuple[str, str, float]]:
    """Return the strongest distinct feature pairs above the configured threshold."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []

    for col_a, row in correlation_matrix.items():
        for col_b, value in (row or {}).items():
            if col_a == col_b:
                continue
            try:
                r = float(value)
            except (TypeError, ValueError):
                continue
            if abs(r) < SETTINGS.CORRELATION_STRONG_THRESHOLD:
                continue
            key = tuple(sorted((str(col_a), str(col_b))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((str(col_a), str(col_b), r))

    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs[: SETTINGS.LLM_MAX_CORRELATIONS]
