"""Dataset profiling module.

Provides the DatasetProfiler class that computes comprehensive
statistical profiles of uploaded datasets.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd

from config.settings import SETTINGS

logger = logging.getLogger(__name__)


def detect_problem_type(df: pd.DataFrame, target_col: str) -> tuple[str, float, str]:
    """Auto-detect the problem type (Classification/Regression/Time Series) for the target.

    This is the single source of truth for task-type inference. The upload page uses
    the full tuple to show a confidence-scored suggestion; the profiler uses only the
    label as a fallback when the user's explicit choice is unavailable.

    Args:
        df: The dataset being profiled.
        target_col: Name of the column the user wants to predict.

    Returns:
        tuple containing (problem_type, confidence_score, explanation_reason)
    """
    col_data = df[target_col]
    dtype_str = str(col_data.dtype)

    # 1. Datetime check -> Time Series
    if "datetime" in dtype_str or "date" in dtype_str or col_data.apply(lambda x: isinstance(x, pd.Timestamp)).all():
        return (
            "Time Series",
            0.90,
            f"Target column '{target_col}' has a datetime dtype ({dtype_str}), suggesting a forecasting task."
        )

    # If it is numeric
    if pd.api.types.is_numeric_dtype(col_data):
        unique_count = col_data.nunique()
        if unique_count <= SETTINGS.MAX_UNIQUE_VALUES_FOR_CLASSIFICATION:
            return (
                "Classification",
                0.85,
                f"Target column '{target_col}' is numeric but has only {unique_count} unique values (<= {SETTINGS.MAX_UNIQUE_VALUES_FOR_CLASSIFICATION}), indicating discrete classes."
            )
        else:
            return (
                "Regression",
                0.90,
                f"Target column '{target_col}' is numeric and has a high number of unique values ({unique_count}), indicating a continuous target."
            )

    # Object / String / Categorical / Bool
    unique_count = col_data.nunique()
    return (
        "Classification",
        0.95,
        f"Target column '{target_col}' is categorical/object type with {unique_count} unique classes, suggesting a classification task."
    )


class DatasetProfiler:
    """Computes a comprehensive statistical profile of a dataset.

    Analyses include: row/column counts, missing values, duplicates,
    dtypes, IQR-based outliers, correlation matrix, cardinality,
    memory usage, class balance, and skewness.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        problem_type: str | None = None,
    ) -> None:
        """Initialise the profiler with a DataFrame, target column, and problem type.

        Args:
            df: The dataset to profile.
            target_column: Name of the target column, if one has been selected.
            problem_type: The user's confirmed task type. When omitted it is inferred
                via ``detect_problem_type`` so the profile always carries a usable value
                for downstream advisors.
        """
        self.df = df
        self.target_column = target_column

        if problem_type is None and target_column and target_column in df.columns:
            problem_type = detect_problem_type(df, target_column)[0]
            logger.info("problem_type not supplied; inferred '%s'.", problem_type)
        self.problem_type = problem_type

        logger.info("DatasetProfiler initialised with shape %s.", df.shape)

    def compute_profile(self) -> dict[str, Any]:
        """Compute and return the full dataset profile.

        Returns:
            Dictionary containing dataset metrics and statistics.
        """
        logger.info("Computing dataset profile...")
        df = self.df
        rows, cols = df.shape
        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = (df.isnull().mean() * 100).to_dict()
        duplicates_count = int(df.duplicated().sum())
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        memory_bytes = int(df.memory_usage(deep=True).sum())

        numeric_cols = []
        categorical_cols = []
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        cardinality = df.nunique().to_dict()

        # Skewness and Outliers for numeric columns
        skewness_dict = {}
        outliers_dict = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                skewness_dict[col] = float(col_data.skew())
                
                # IQR Outlier Detection
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - SETTINGS.OUTLIER_IQR_MULTIPLIER * iqr
                upper_bound = q3 + SETTINGS.OUTLIER_IQR_MULTIPLIER * iqr
                
                outlier_indices = col_data[(col_data < lower_bound) | (col_data > upper_bound)].index.tolist()
                outliers_dict[col] = {
                    "count": len(outlier_indices),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }
            else:
                skewness_dict[col] = 0.0
                outliers_dict[col] = {"count": 0, "lower_bound": 0.0, "upper_bound": 0.0}

        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr(method="pearson")
            # Replace NaN with 0 for plotting/advising safety
            corr_df = corr_df.fillna(0.0)
            correlation_matrix = corr_df.to_dict()
        else:
            correlation_matrix = {}

        # Class balance (classification only — a continuous target would otherwise
        # produce a value_counts dict with one entry per distinct value, which is
        # meaningless as a class distribution and is what previously made every
        # advisor treat regression datasets as classification).
        class_balance = {}
        target_summary = {}
        if self.target_column and self.target_column in df.columns:
            target_data = df[self.target_column]
            if self.problem_type == "Classification":
                class_balance = target_data.value_counts(dropna=False).to_dict()

            # Simple summary stats for target
            if pd.api.types.is_numeric_dtype(target_data):
                target_summary = {
                    "mean": float(target_data.mean()) if not target_data.isnull().all() else 0.0,
                    "median": float(target_data.median()) if not target_data.isnull().all() else 0.0,
                    "min": float(target_data.min()) if not target_data.isnull().all() else 0.0,
                    "max": float(target_data.max()) if not target_data.isnull().all() else 0.0,
                }

        profile = {
            "problem_type": self.problem_type,
            "shape": (rows, cols),
            "missing_values": missing_counts,
            "missing_pct": missing_pcts,
            "duplicates": duplicates_count,
            "dtypes": dtypes_dict,
            "cardinality": cardinality,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "skewness": skewness_dict,
            "outliers": outliers_dict,
            "correlation_matrix": correlation_matrix,
            "memory_bytes": memory_bytes,
            "class_balance": class_balance,
            "target_summary": target_summary,
        }

        logger.info("Dataset profile computation complete.")
        return profile
