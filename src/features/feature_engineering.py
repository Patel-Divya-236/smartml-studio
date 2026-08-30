"""Feature engineering pipeline.

Applies the user's opt-in feature engineering steps as a *fittable* transformer.
Every step that learns from data — variance thresholds, polynomial expansion,
PCA components, and above all SelectKBest — is fitted on the training rows only.
SelectKBest is the sharpest of these: fitting it on the full dataset chooses
features using the test set's target values, which is direct label leakage.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """Fit-once / transform-many feature engineering.

    Steps run in a fixed order — low-variance filter, polynomial expansion, PCA,
    then SelectKBest — each one enabled by a flag in the config dictionary.

    Attributes:
        config: Flags and parameters chosen on the Feature Engineering page.
        problem_type: Drives the SelectKBest score function.
    """

    def __init__(self, config: dict[str, Any], problem_type: str = "Classification") -> None:
        """Initialise the pipeline with the user's configuration."""
        self.config = config or {}
        self.problem_type = problem_type

        # Fitted state, populated by fit()
        self._low_variance_kept: list[str] | None = None
        self._poly: PolynomialFeatures | None = None
        self._poly_input_columns: list[str] = []
        self._pca: PCA | None = None
        self._pca_input_columns: list[str] = []
        self._selected_columns: list[str] | None = None
        self._feature_names_out: list[str] = []
        self._is_fitted = False

        logger.info(
            "FeatureEngineeringPipeline initialised (problem_type=%s, steps=%s).",
            problem_type,
            [k for k, v in self.config.items() if k.endswith("_active") and v],
        )

    # ── Public API ────────────────────────────────────────────────────

    @property
    def feature_names_out(self) -> list[str]:
        """Column names produced by ``transform``, in order."""
        self._check_fitted()
        return list(self._feature_names_out)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FeatureEngineeringPipeline":
        """Learn every step from the training rows only.

        Args:
            X_train: Training features (target already removed).
            y_train: Training targets, used by SelectKBest.

        Returns:
            self, for chaining.
        """
        logger.info("Fitting FeatureEngineeringPipeline on %d training rows.", len(X_train))
        self._reset_fitted_state()

        X = X_train.copy()

        # ── 1. Low Variance Filter ──────────────────────────────────
        if self.config.get("low_variance_active", False):
            threshold = self.config.get("low_variance_threshold", 0.01)
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
            if numeric_cols:
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(X[numeric_cols])
                kept = [c for c, keep in zip(numeric_cols, selector.get_support()) if keep]
                dropped = [c for c in numeric_cols if c not in kept]
                # Non-numeric columns are never candidates and always survive.
                self._low_variance_kept = [c for c in X.columns if c not in dropped]
                if dropped:
                    logger.info("Low-variance filter drops: %s", dropped)
                X = X[self._low_variance_kept]

        # ── 2. Polynomial Features ──────────────────────────────────
        if self.config.get("poly_active", False):
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
            if numeric_cols:
                self._poly = PolynomialFeatures(
                    degree=self.config.get("poly_degree", 2),
                    interaction_only=self.config.get("poly_interaction_only", False),
                    include_bias=False,
                )
                self._poly.fit(X[numeric_cols])
                self._poly_input_columns = numeric_cols
                X = self._apply_poly(X)

        # ── 3. PCA ──────────────────────────────────────────────────
        if self.config.get("pca_active", False):
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
            if numeric_cols:
                n_components = min(self.config.get("pca_components", 2), len(numeric_cols), len(X))
                self._pca = PCA(n_components=n_components)
                self._pca.fit(X[numeric_cols])
                self._pca_input_columns = numeric_cols
                X = self._apply_pca(X)

        # ── 4. Feature Selection ────────────────────────────────────
        if self.config.get("select_k_best_active", False):
            k = min(self.config.get("select_k_best_k", 5), X.shape[1])
            selector = SelectKBest(score_func=self._score_func(), k=k)
            # y_train only — this is the step that would otherwise leak labels.
            selector.fit(X, y_train)
            self._selected_columns = [c for c, keep in zip(X.columns, selector.get_support()) if keep]
            logger.info("SelectKBest kept %d features: %s", k, self._selected_columns)
            X = X[self._selected_columns]

        self._is_fitted = True
        self._feature_names_out = list(X.columns)
        logger.info(
            "FeatureEngineeringPipeline fitted: %d input columns -> %d output columns.",
            X_train.shape[1],
            len(self._feature_names_out),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replay the fitted steps on a new frame."""
        self._check_fitted()
        out = X.copy()

        if self._low_variance_kept is not None:
            out = out[self._low_variance_kept]
        if self._poly is not None:
            out = self._apply_poly(out)
        if self._pca is not None:
            out = self._apply_pca(out)
        if self._selected_columns is not None:
            out = out[self._selected_columns]

        return out.reindex(columns=self._feature_names_out)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit on the training rows and return the transformed training frame."""
        return self.fit(X_train, y_train).transform(X_train)

    # ── Internal helpers ──────────────────────────────────────────────

    def _reset_fitted_state(self) -> None:
        """Clear fitted state so a pipeline can be safely re-fitted."""
        self._low_variance_kept = None
        self._poly = None
        self._poly_input_columns = []
        self._pca = None
        self._pca_input_columns = []
        self._selected_columns = None
        self._feature_names_out = []
        self._is_fitted = False

    def _check_fitted(self) -> None:
        """Raise if the pipeline has not been fitted yet."""
        if not self._is_fitted:
            raise ValueError("FeatureEngineeringPipeline must be fitted before use. Call fit() first.")

    def _score_func(self):
        """Return the SelectKBest score function for the current problem type."""
        is_classification = self.problem_type == "Classification"
        use_anova = self.config.get("select_k_best_method", "ANOVA") == "ANOVA"
        if is_classification:
            return f_classif if use_anova else mutual_info_classif
        return f_regression if use_anova else mutual_info_regression

    def _apply_poly(self, X: pd.DataFrame) -> pd.DataFrame:
        """Expand the fitted numeric columns, keeping non-numeric columns alongside."""
        expanded = self._poly.transform(X[self._poly_input_columns])
        names = self._poly.get_feature_names_out(self._poly_input_columns)
        X_poly = pd.DataFrame(expanded, columns=names, index=X.index)

        passthrough = [c for c in X.columns if c not in self._poly_input_columns]
        if passthrough:
            return pd.concat([X_poly, X[passthrough]], axis=1)
        return X_poly

    def _apply_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project the fitted numeric columns, keeping non-numeric columns alongside."""
        components = self._pca.transform(X[self._pca_input_columns])
        names = [f"PC{i + 1}" for i in range(components.shape[1])]
        X_pca = pd.DataFrame(components, columns=names, index=X.index)

        passthrough = [c for c in X.columns if c not in self._pca_input_columns]
        if passthrough:
            return pd.concat([X_pca, X[passthrough]], axis=1)
        return X_pca
