"""Dataset upload, target selection, and profiling."""

import io
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.api.deps import get_session, require
from backend.core.serialization import dataframe_to_records, to_jsonable
from backend.core.session import Session
from src.profiling.dataset_profiler import DatasetProfiler, detect_problem_type

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # matches the 200MB limit shown in the UI
PREVIEW_ROWS = 50


class TargetSelection(BaseModel):
    """Target column and, optionally, an override of the detected task type."""

    target_column: str
    problem_type: str | None = None


@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict:
    """Read a CSV or Excel upload into the session and return a preview.

    Uploading invalidates every downstream stage — the whole point of the cascade — so a
    second upload cannot leave metrics from the first one on screen.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 200MB limit.")

    name = (file.filename or "dataset.csv").strip()
    try:
        if name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        logger.exception("Failed to parse upload %s", name)
        raise HTTPException(status_code=422, detail=f"Could not read the file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=422, detail="The uploaded file has no rows.")

    session.reset_downstream("dataset")
    session.set("dataset", df)
    session.set("dataset_name", name)

    logger.info("Uploaded %s: %d rows, %d columns.", name, len(df), df.shape[1])
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 2),
        "preview": dataframe_to_records(df, limit=PREVIEW_ROWS),
        "completed_steps": session.completed_steps(),
    }


@router.get("/columns")
def list_columns(session: Session = Depends(get_session)) -> dict:
    """List candidate target columns with the task type each would imply."""
    df: pd.DataFrame = require(session, "dataset", "upload")

    suggestions = []
    for col in df.columns:
        try:
            problem_type, confidence, rationale = detect_problem_type(df, str(col))
        except Exception:
            problem_type, confidence, rationale = "Classification", 0.0, "Undetermined."
        suggestions.append({
            "column": str(col),
            "dtype": str(df[col].dtype),
            "unique": int(df[col].nunique(dropna=True)),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "problem_type": problem_type,
            "confidence": to_jsonable(confidence),
            "rationale": rationale,
        })
    return {"columns": suggestions}


@router.post("/target")
def set_target(
    payload: TargetSelection,
    session: Session = Depends(get_session),
) -> dict:
    """Set the target column and task type, invalidating everything downstream."""
    df: pd.DataFrame = require(session, "dataset", "upload")

    if payload.target_column not in df.columns:
        raise HTTPException(status_code=422, detail="That column is not in the dataset.")

    detected, confidence, rationale = detect_problem_type(df, payload.target_column)
    problem_type = payload.problem_type or detected

    session.reset_downstream("dataset_name")
    session.set("target_column", payload.target_column)
    session.set("problem_type", problem_type)

    return {
        "target_column": payload.target_column,
        "problem_type": problem_type,
        "detected_problem_type": detected,
        "detection_confidence": to_jsonable(confidence),
        "detection_rationale": rationale,
        "warnings": _target_warnings(df, payload.target_column, problem_type),
        "completed_steps": session.completed_steps(),
    }


@router.get("/profile")
def get_profile(session: Session = Depends(get_session)) -> dict:
    """Compute (and cache) the full dataset profile."""
    df: pd.DataFrame = require(session, "dataset", "upload")
    target = session.get("target_column")
    if target is None:
        raise HTTPException(status_code=409, detail="Choose a target column first.")

    cached = session.get("profile")
    if cached is None:
        profiler = DatasetProfiler(df, target_column=target, problem_type=session.get("problem_type"))
        cached = profiler.compute_profile()
        session.set("profile", cached)
        logger.info("Profile computed for %s.", session.get("dataset_name"))

    rows, cols = cached.get("shape", (0, 0))
    missing_total = int(sum(cached.get("missing_values", {}).values()))

    return {
        "profile": to_jsonable(cached),
        "summary": {
            "rows": int(rows),
            "columns": int(cols),
            "duplicates": int(cached.get("duplicates", 0)),
            "duplicate_pct": round(float(cached.get("duplicates", 0)) / max(rows, 1) * 100, 2),
            "missing_total": missing_total,
            "missing_pct": round(missing_total / max(rows * cols, 1) * 100, 2),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 2),
            "problem_type": session.get("problem_type"),
            "target_column": target,
            "n_classes": len(cached.get("class_balance", {}) or {}),
        },
        "observations": _key_observations(cached),
        "completed_steps": session.completed_steps(),
    }



@router.get("/distribution/{column}")
def column_distribution(
    column: str,
    bins: int = 20,
    top: int = 25,
    session: Session = Depends(get_session),
) -> dict:
    """Return a plottable distribution for one column.

    Categorical columns come back as value counts, numeric columns as histogram bins.
    The profile deliberately stores only aggregates, so this is computed on demand from
    the uploaded frame rather than cached into the profile -- and, like everything else
    the API returns, it is counts rather than rows.
    """
    df: pd.DataFrame = require(session, "dataset", "upload")
    if column not in df.columns:
        raise HTTPException(status_code=404, detail="That column is not in the dataset.")

    series = df[column].dropna()
    if series.empty:
        return {"column": column, "kind": "empty", "data": []}

    is_numeric = pd.api.types.is_numeric_dtype(series)
    # A numeric column with very few distinct values is a coded category, not a
    # measurement; binning it into 20 buckets would produce a misleading picture.
    if is_numeric and series.nunique() > 20:
        counts, edges = np.histogram(series.astype(float), bins=min(bins, 50))
        return {
            "column": column,
            "kind": "histogram",
            "data": [
                {
                    "name": f"{edges[i]:.4g} – {edges[i + 1]:.4g}",
                    "midpoint": float((edges[i] + edges[i + 1]) / 2),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ],
        }

    counts = series.astype(str).value_counts()
    total = int(counts.sum())
    head = counts.head(top)
    data = [
        {"name": str(name), "count": int(value), "share": round(value / total * 100, 2)}
        for name, value in head.items()
    ]

    # Free text is not a category. Long labels on a bar chart overlap into an unreadable
    # smear, and near-unique values make the bars all the same height, so the chart
    # carries no information at all. Say so instead of drawing it.
    avg_label_length = float(head.index.to_series().astype(str).str.len().mean()) if len(head) else 0.0
    distinct = int(counts.size)
    is_free_text = avg_label_length > 40 or (len(series) and distinct / len(series) > 0.5)

    return {
        "column": column,
        "kind": "free_text" if is_free_text else "categories",
        "distinct": distinct,
        "rows": int(len(series)),
        "avg_label_length": round(avg_label_length, 1),
        "truncated": distinct > len(data),
        "data": data,
    }


def _target_warnings(df: pd.DataFrame, column: str, problem_type: str) -> list[dict]:
    """Flag target columns that will not train usefully.

    A classification target with almost as many distinct values as rows is not a set of
    classes — it is an identifier or free text. sklearn and LightGBM only warn about this
    deep inside training, by which point the user has already waited through a fit, so it
    is reported here instead, at the point the choice is made.
    """
    warnings: list[dict] = []
    series = df[column]
    rows = len(series)
    distinct = int(series.nunique(dropna=True))
    missing_pct = float(series.isna().mean() * 100)

    if rows == 0:
        return warnings

    ratio = distinct / rows

    if problem_type == "Classification":
        if ratio > 0.5:
            warnings.append({
                "severity": "danger",
                "text": (
                    f"'{column}' has {distinct:,} distinct values across {rows:,} rows "
                    f"({ratio * 100:.0f}%). That is an identifier or free text, not a set "
                    "of classes — training will produce meaningless accuracy."
                ),
            })
        elif distinct > 50:
            warnings.append({
                "severity": "warning",
                "text": (
                    f"'{column}' has {distinct:,} classes. Models will train, but per-class "
                    "metrics get thin and confusion matrices become hard to read."
                ),
            })
        elif distinct < 2:
            warnings.append({
                "severity": "danger",
                "text": f"'{column}' has only one distinct value, so there is nothing to predict.",
            })

    if missing_pct > 20:
        warnings.append({
            "severity": "warning",
            "text": (
                f"{missing_pct:.1f}% of '{column}' is missing. Those rows are dropped before "
                "the split, so the usable dataset is smaller than it looks."
            ),
        })

    avg_length = 0.0
    # Not `dtype == object`: pandas reports a text column as `str` under the newer string
    # dtype, so that comparison is False and silently skips the check.
    if not pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().astype(str).head(500)
        if len(sample):
            avg_length = float(sample.str.len().mean())
    if avg_length > 60:
        warnings.append({
            "severity": "warning",
            "text": (
                f"Values in '{column}' average {avg_length:.0f} characters. Long free text "
                "does not work as a category — charts and labels will be unreadable."
            ),
        })

    return warnings

def _key_observations(profile: dict) -> list[dict]:
    """Derive the short 'Key Observations' list the analysis page shows.

    Severity drives the badge colour in the UI, so it is decided here rather than being
    inferred from wording on the client.
    """
    out: list[dict] = []
    rows, cols = profile.get("shape", (0, 0))

    duplicates = int(profile.get("duplicates", 0))
    if duplicates:
        out.append({
            "severity": "warning",
            "text": f"Found {duplicates:,} duplicate rows. Consider dropping them in preprocessing.",
        })

    missing_total = int(sum(profile.get("missing_values", {}).values()))
    if missing_total:
        affected = sum(1 for v in profile.get("missing_values", {}).values() if v)
        out.append({
            "severity": "warning",
            "text": f"Dataset contains {missing_total:,} missing values across {affected} column(s).",
        })

    balance = profile.get("class_balance") or {}
    if balance:
        counts = list(balance.values())
        ratio = max(counts) / max(min(counts), 1)
        severity = "warning" if ratio > 3 else "success"
        out.append({
            "severity": severity,
            "text": f"Target has {len(balance)} classes; largest is {ratio:.1f}x the smallest.",
        })

    cardinality = profile.get("cardinality", {}) or {}
    high_card = [c for c, v in cardinality.items() if rows and v > max(50, rows * 0.5)]
    if high_card:
        out.append({
            "severity": "info",
            "text": f"{len(high_card)} column(s) have very high cardinality: {', '.join(map(str, high_card[:3]))}.",
        })

    if not out:
        out.append({"severity": "success", "text": "No structural data-quality issues detected."})
    return out
