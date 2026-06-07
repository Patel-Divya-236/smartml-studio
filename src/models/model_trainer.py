"""Model training orchestrator.

Coordinates training of selected models, tracks timing,
and stores results in session state.
"""

import logging
import time
from typing import Any

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
            return RandomForestClassifier(random_state=42) if is_class else RandomForestRegressor(random_state=42)
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
            return SVC(probability=True, random_state=42) if is_class else SVR()
        elif name == "KNN":
            return KNeighborsClassifier() if is_class else KNeighborsRegressor()
        elif name == "Custom SVM":
            if is_class:
                return CustomSVM(kernel="linear", C=1.0, learning_rate=0.01, n_iters=500)
            raise ValueError("Custom SVM from scratch is currently only implemented for Classification.")
        elif name == "Custom KNN":
            if is_class:
                return CustomKNN(n_neighbors=5, metric="euclidean")
            raise ValueError("Custom KNN from scratch is currently only implemented for Classification.")
        else:
            raise ValueError(f"Unknown model name: {name}")

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        selected_models: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Train the selected models and collect performance metrics.

        Returns:
            Dictionary containing model metrics and trained model instances.
        """
        results = {}
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
            except Exception as e:
                logger.error("Failed to train model %s: %s", name, str(e), exc_info=True)
                raise e

        return results
