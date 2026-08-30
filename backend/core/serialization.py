"""JSON-safe conversion for pandas and numpy values.

FastAPI's default encoder raises on `np.int64`, and `float('nan')` serialises to the
literal `NaN`, which is invalid JSON and silently breaks `JSON.parse` in the browser.
Both are pervasive in profile dictionaries and metric tables, so every response passes
through `to_jsonable` before it leaves a handler.
"""

import math
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python primitives.

    Non-finite floats become None rather than NaN/Infinity, which are not valid JSON.
    """
    if value is None:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    # Plain Python ints must be listed explicitly. Without them an int falls past every
    # branch below and is stringified by the final `str(value)`, which turns counts into
    # "79" and silently breaks any chart that plots them.
    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None

    if isinstance(value, (np.str_, str)):
        return str(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]

    if isinstance(value, pd.Series):
        return [to_jsonable(v) for v in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return dataframe_to_records(value)

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if hasattr(value, "item"):  # remaining numpy scalars
        try:
            return to_jsonable(value.item())
        except (ValueError, AttributeError):
            pass

    return str(value)


def dataframe_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-safe row dictionaries."""
    frame = df.head(limit) if limit is not None else df
    return [
        {str(col): to_jsonable(row[col]) for col in frame.columns}
        for _, row in frame.iterrows()
    ]


def recommendation_to_dict(rec: Any) -> dict[str, Any]:
    """Serialise a Recommendation dataclass for the API.

    The fields map one-to-one onto what `RecommendationCard` renders, so the front end
    needs no translation layer.
    """
    return {
        "label": rec.label,
        "confidence_score": to_jsonable(rec.confidence_score),
        "star_rating": int(rec.star_rating),
        "reason": rec.reason,
        "why_explanation": rec.why_explanation,
        "category": rec.category,
        "metadata": to_jsonable(rec.metadata),
    }
