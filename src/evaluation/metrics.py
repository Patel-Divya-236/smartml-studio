"""Model comparison metrics and feature importance extraction.

Moved out of `pages/8_model_comparison.py` during the React/FastAPI migration: these are
domain calculations, not view code, and the API needs them without importing Streamlit.
Behaviour is unchanged from the Streamlit implementation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    trained_models: dict,
    y_test: np.ndarray,
    problem_type: str,
) -> pd.DataFrame:
    """Compute performance metrics for all trained models."""
    logger.info("Computing evaluation metrics for trained models...")
    rows = []

    for name, res in trained_models.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        fit_time = res["fit_time"]
        predict_time = res["predict_time"]

        if problem_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            # Macro average handles multi-class labels properly.
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )

            auc = np.nan
            if y_prob is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        prob_pos = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
                        auc = roc_auc_score(y_test, prob_pos)
                    else:
                        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except Exception as e:
                    logger.debug("Could not compute ROC-AUC for %s: %s", name, str(e))

            rows.append({
                "Model Name": name,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "Fit Time (s)": fit_time,
                "Predict Time (s)": predict_time,
            })
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            rows.append({
                "Model Name": name,
                "R² Score": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "Fit Time (s)": fit_time,
                "Predict Time (s)": predict_time,
            })

    comparison_df = pd.DataFrame(rows)
    logger.info("Comparison metrics computed.")
    return comparison_df


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract and sort feature importance coefficients/weights from a model.

    Returns None when the estimator exposes neither `feature_importances_` nor `coef_`,
    or when the width does not match the feature list.
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.mean(np.abs(coef), axis=0) if len(coef.shape) == 2 else np.abs(coef)

    if importances is not None and len(importances) == len(feature_names):
        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

    return None
