"""Preprocessing pipeline.

Applies the user's per-column imputation, scaling, and encoding plan as a
*fittable* transformer: every statistic (means, medians, quartiles, category
vocabularies) is learned from the training rows alone and then replayed onto the
test rows. Fitting on the full dataset — as an earlier version of this pipeline
did — leaks test-set information into training and inflates every reported metric.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Fit-once / transform-many preprocessing for a per-column strategy plan.

    The three config dictionaries map a column name to the action chosen by the
    user on the Preprocessing page. Unknown or ``"None"`` actions leave the column
    untouched.

    Attributes:
        target_column: Name of the target column, excluded from all transformations.
        impute_config: column -> one of None/Mean/Median/Mode/Most Frequent/KNN/Drop Column.
        encode_config: column -> one of None/One-Hot/Ordinal-Label.
        scale_config: column -> one of None/Standard/MinMax/Robust/Log1p.
    """

    #: Imputation actions that remove the column outright rather than filling it.
    DROP_ACTION = "Drop Column"

    def __init__(
        self,
        target_column: str,
        impute_config: dict[str, str],
        encode_config: dict[str, str],
        scale_config: dict[str, str],
    ) -> None:
        """Initialise the pipeline with the user's per-column plan."""
        self.target_column = target_column
        self.impute_config = impute_config or {}
        self.encode_config = encode_config or {}
        self.scale_config = scale_config or {}

        # Fitted state, populated by fit()
        self._imputers: dict[str, object] = {}
        self._scalers: dict[str, object] = {}
        self._encoders: dict[str, object] = {}
        self._log1p_shifts: dict[str, float] = {}
        self._dropped_columns: list[str] = []
        self._input_columns: list[str] = []
        self._feature_names_out: list[str] = []
        self._is_fitted = False

        logger.info(
            "PreprocessingPipeline initialised (target=%s, %d imputation / %d encoding / %d scaling rules).",
            target_column,
            len(self.impute_config),
            len(self.encode_config),
            len(self.scale_config),
        )

    # ── Public API ────────────────────────────────────────────────────

    @property
    def feature_names_out(self) -> list[str]:
        """Column names produced by ``transform``, in order."""
        self._check_fitted()
        return list(self._feature_names_out)

    @property
    def dropped_columns(self) -> list[str]:
        """Columns removed because the user selected 'Drop Column'."""
        return list(self._dropped_columns)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "PreprocessingPipeline":
        """Learn every transformer from the training rows only.

        Args:
            X_train: Training features. Must not contain the target column.
            y_train: Unused; accepted for interface symmetry with sklearn.

        Returns:
            self, for chaining.
        """
        logger.info("Fitting PreprocessingPipeline on %d training rows.", len(X_train))
        self._reset_fitted_state()

        working = self._drop_configured_columns(X_train)
        self._input_columns = list(working.columns)

        for col in working.columns:
            series = working[[col]]

            # 1. Imputation — fitted on train so fill values are train statistics.
            imputer = self._make_imputer(self.impute_config.get(col, "None"))
            if imputer is not None:
                imputer.fit(series)
                self._imputers[col] = imputer
                series = self._apply_imputer(imputer, series, col)

            # 2. Scaling / transformation, then 3. encoding — same order as before,
            #    so per-column behaviour is unchanged relative to the previous pipeline.
            series = self._fit_scaler(col, series)
            self._fit_encoder(col, series)

        self._is_fitted = True
        # Establish the output column order by transforming the training frame once.
        self._feature_names_out = list(self._transform_frame(working).columns)
        logger.info(
            "PreprocessingPipeline fitted: %d input columns -> %d output columns.",
            len(self._input_columns),
            len(self._feature_names_out),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformers to a new frame.

        Args:
            X: Features to transform. Must contain the columns seen during ``fit``.

        Returns:
            Transformed features with exactly ``feature_names_out`` columns, in order.
        """
        self._check_fitted()
        working = self._drop_configured_columns(X)

        missing = [c for c in self._input_columns if c not in working.columns]
        if missing:
            raise ValueError(f"Frame is missing columns seen during fit: {missing}")

        transformed = self._transform_frame(working[self._input_columns])
        # Reindex so train and test always agree on columns and order, even if an
        # encoder produced a different set for an unseen category distribution.
        return transformed.reindex(columns=self._feature_names_out, fill_value=0.0)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> pd.DataFrame:
        """Fit on the training rows and return the transformed training frame."""
        return self.fit(X_train, y_train).transform(X_train)

    # ── Internal helpers ──────────────────────────────────────────────

    def _reset_fitted_state(self) -> None:
        """Clear fitted state so a pipeline can be safely re-fitted."""
        self._imputers.clear()
        self._scalers.clear()
        self._encoders.clear()
        self._log1p_shifts.clear()
        self._dropped_columns = []
        self._input_columns = []
        self._feature_names_out = []
        self._is_fitted = False

    def _check_fitted(self) -> None:
        """Raise if the pipeline has not been fitted yet."""
        if not self._is_fitted:
            raise ValueError("PreprocessingPipeline must be fitted before use. Call fit() first.")

    def _drop_configured_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X without the target and any 'Drop Column' selections."""
        working = X.copy()

        if self.target_column in working.columns:
            working = working.drop(columns=[self.target_column])

        to_drop = [
            col
            for col, action in self.impute_config.items()
            if action == self.DROP_ACTION and col in working.columns and col != self.target_column
        ]
        if to_drop:
            working = working.drop(columns=to_drop)
            if not self._dropped_columns:
                self._dropped_columns = sorted(to_drop)
                logger.info("Dropping columns per user config: %s", self._dropped_columns)

        return working

    @staticmethod
    def _make_imputer(action: str) -> object | None:
        """Return an unfitted imputer for the given action, or None."""
        if action == "Mean":
            return SimpleImputer(strategy="mean")
        if action == "Median":
            return SimpleImputer(strategy="median")
        if action in ("Mode", "Most Frequent"):
            return SimpleImputer(strategy="most_frequent")
        if action == "KNN":
            return KNNImputer(n_neighbors=5)
        return None

    @staticmethod
    def _apply_imputer(imputer: object, series: pd.DataFrame, col: str) -> pd.DataFrame:
        """Transform a single-column frame with a fitted imputer, preserving labels."""
        filled = imputer.transform(series)
        return pd.DataFrame(filled, columns=[col], index=series.index)

    def _fit_scaler(self, col: str, series: pd.DataFrame) -> pd.DataFrame:
        """Fit the configured scaler for one column and return the scaled series.

        The scaled output is returned because the encoder (fitted next) must see the
        same values ``transform`` will hand it.
        """
        action = self.scale_config.get(col, "None")
        if action == "None":
            return series

        if action == "Log1p":
            self._log1p_shifts[col] = self._fit_log1p_shift(series, col)
            series = self._apply_log1p(series, col, self._log1p_shifts[col])
            scaler = StandardScaler()
        elif action == "Standard":
            scaler = StandardScaler()
        elif action == "MinMax":
            scaler = MinMaxScaler()
        elif action == "Robust":
            scaler = RobustScaler()
        else:
            return series

        scaler.fit(series)
        self._scalers[col] = scaler
        return pd.DataFrame(scaler.transform(series), columns=[col], index=series.index)

    def _fit_encoder(self, col: str, series: pd.DataFrame) -> None:
        """Fit the configured encoder for one column on the training values."""
        action = self.encode_config.get(col, "None")
        if action == "One-Hot":
            encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        elif action == "Ordinal/Label":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            return

        encoder.fit(series.astype(str))
        self._encoders[col] = encoder

    def _transform_frame(self, working: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transformers column by column and reassemble."""
        parts: list[pd.DataFrame] = []

        for col in working.columns:
            series = working[[col]]

            imputer = self._imputers.get(col)
            if imputer is not None:
                series = self._apply_imputer(imputer, series, col)

            if col in self._log1p_shifts:
                series = self._apply_log1p(series, col, self._log1p_shifts[col])

            scaler = self._scalers.get(col)
            if scaler is not None:
                series = pd.DataFrame(scaler.transform(series), columns=[col], index=series.index)

            encoder = self._encoders.get(col)
            if encoder is not None:
                encoded = encoder.transform(series.astype(str))
                if isinstance(encoder, OneHotEncoder):
                    names = encoder.get_feature_names_out([col])
                else:
                    names = [col]
                series = pd.DataFrame(encoded, columns=names, index=series.index)

            parts.append(series)

        if not parts:
            return pd.DataFrame(index=working.index)
        return pd.concat(parts, axis=1)

    @staticmethod
    def _fit_log1p_shift(series: pd.DataFrame, col: str) -> float:
        """Learn the constant offset needed to keep log1p defined on this column.

        Raw ``log1p`` on values at or below -1 yields NaN or -inf, which silently
        poisons every downstream model. The shift is learned from the training rows
        and reused verbatim at transform time so train and test stay on one scale.
        """
        minimum = pd.to_numeric(series[col], errors="coerce").min()
        if pd.notna(minimum) and minimum <= -1:
            shift = float(abs(minimum)) + 1.0
            logger.warning(
                "Column '%s' contains values <= -1; shifting by %.4f before log1p to avoid NaN.",
                col,
                shift,
            )
            return shift
        return 0.0

    @staticmethod
    def _apply_log1p(series: pd.DataFrame, col: str, shift: float) -> pd.DataFrame:
        """Apply log1p using the shift learned during fit."""
        numeric = pd.to_numeric(series[col], errors="coerce") + shift
        # Test rows can still fall below the training minimum; clip rather than emit NaN.
        numeric = numeric.clip(lower=-1 + 1e-9)
        return pd.DataFrame(np.log1p(numeric), columns=[col], index=series.index)
