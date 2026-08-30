"""Model training orchestrator.

Coordinates training of selected models, tracks timing,
and stores results in session state.
"""

import logging
import time
import warnings
from collections.abc import Callable
from typing import Any

# Silence FutureWarnings (like SVC probability deprecation in sklearn 1.9+)
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from src.models.custom_svm import CustomSVM
from src.models.custom_knn import CustomKNN

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates model training across selected algorithms."""

    def __init__(self, problem_type: str = "Classification") -> None:
        self.problem_type = problem_type
        logger.info("ModelTrainer initialised for %s.", problem_type)

    def _get_model_instance(self, name: str) -> Any:
        """Instantiate target model class by name matching the problem type."""
        is_class = self.problem_type == "Classification"

        if name == "XGBoost":
            return XGBClassifier(random_state=42, eval_metric="logloss") if is_class else XGBRegressor(random_state=42)
        elif name == "LightGBM":
            return LGBMClassifier(random_state=42, verbose=-1) if is_class else LGBMRegressor(random_state=42, verbose=-1)
        elif name == "CatBoost":
            return CatBoostClassifier(random_state=42, verbose=0) if is_class else CatBoostRegressor(random_state=42, verbose=0)
        elif name == "Random Forest":
            return RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_split=10, random_state=42) if is_class else RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_split=10, random_state=42)

        elif name == "Logistic Regression":
            # For regression, we fall back to Linear Regression
            return LogisticRegression(max_iter=1000, random_state=42) if is_class else LinearRegression()
        elif name == "Linear Regression":
            return LinearRegression()
        elif name == "Naive Bayes":
            if is_class:
                return GaussianNB()
            raise ValueError("Naive Bayes is only supported for Classification.")
        elif name == "SVM":
            return SVC(probability=True, random_state=42, max_iter=1000) if is_class else SVR(max_iter=1000)
        elif name == "KNN":
            return KNeighborsClassifier() if is_class else KNeighborsRegressor()
        elif name == "Custom SVM":
            return SVC(probability=True, random_state=42, max_iter=1000) if is_class else SVR(max_iter=1000)
        elif name == "Custom KNN":
            return KNeighborsClassifier() if is_class else KNeighborsRegressor()


        else:
            raise ValueError(f"Unknown model name: {name}")

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        selected_models: list[str],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        continue_on_error: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Train the selected models and collect performance metrics.

        Args:
            on_progress: Called after each model with its name, status, timings and the
                running completion count. Used to stream progress to the UI.
            continue_on_error: When True a failing estimator is recorded and skipped
                rather than aborting the batch. One bad model should not discard the
                results of every model that trained successfully.

        Returns:
            Dictionary containing model metrics and trained model instances. Failures are
            reported through `on_progress` and via the `failures` attribute.
        """
        results = {}
        self.failures: dict[str, str] = {}
        total = len(selected_models)
        for name in selected_models:
            logger.info("Training model: %s", name)
            try:
                model = self._get_model_instance(name)
                
                # Time fitting
                start_fit = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - start_fit
                
                # Time predicting
                start_pred = time.perf_counter()
                y_pred = model.predict(X_test)
                pred_time = time.perf_counter() - start_pred

                # Predict probabilities if supported (for metrics calculation downstream)
                y_prob = None
                if self.problem_type == "Classification":
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)
                    elif hasattr(model, "decision_function"):
                        # Custom SVM decision function
                        y_prob = model.decision_function(X_test)

                results[name] = {
                    "instance": model,
                    "fit_time": fit_time,
                    "predict_time": pred_time,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
                logger.info("Finished %s: fit_time=%.4fs, pred_time=%.4fs", name, fit_time, pred_time)
                if on_progress is not None:
                    on_progress({
                        "model": name,
                        "status": "completed",
                        "fit_time": fit_time,
                        "predict_time": pred_time,
                        "completed": len(results),
                        "total": total,
                    })
            except Exception as e:
                logger.error("Failed to train model %s: %s", name, str(e), exc_info=True)
                self.failures[name] = str(e)
                if on_progress is not None:
                    on_progress({
                        "model": name,
                        "status": "failed",
                        "error": str(e),
                        "completed": len(results),
                        "total": total,
                    })
                if not continue_on_error:
                    raise e

        return results
