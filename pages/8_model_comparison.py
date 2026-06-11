"""SmartML Studio — Model Comparison page."""

import logging
from typing import Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

from config.logging_config import setup_logging
from utils.session_state import get_state, set_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def compute_metrics(
    trained_models: dict,
    y_test: np.ndarray,
    problem_type: str
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
            # Use macro average to handle multi-class labels properly
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            
            # Compute ROC-AUC if probability or decision function scores are available
            auc = np.nan
            if y_prob is not None:
                try:
                    # Binary check
                    if len(np.unique(y_test)) == 2:
                        # y_prob might be 2D array or 1D array of positive class probabilities
                        if len(y_prob.shape) == 2:
                            prob_pos = y_prob[:, 1]
                        else:
                            prob_pos = y_prob
                        auc = roc_auc_score(y_test, prob_pos)
                    else:
                        # Multiclass AUC
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
                "Predict Time (s)": predict_time
            })
        else:
            # Regression metrics
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
                "Predict Time (s)": predict_time
            })

    comparison_df = pd.DataFrame(rows)
    logger.info("Comparison metrics computed.")
    return comparison_df


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Extract and sort feature importance coefficients/weights from a model."""
    importances = None
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if len(coef.shape) == 2:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
            
    if importances is not None and len(importances) == len(feature_names):
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        return df_imp
        
    return None


def render_model_comparison_page() -> None:
    """Render the Model Comparison page."""
    st.header("Model Comparison")

    trained_models = get_state("trained_models")
    y_test = get_state("y_test")
    feature_names = get_state("feature_names")
    problem_type = get_state("problem_type")

    # Prerequisite check
    if trained_models is None or y_test is None:
        st.warning("Please train models first in the Model Training page.")
        st.stop()


    # Calculate or retrieve comparison table
    comparison_df = get_state("model_comparison")
    if comparison_df is None:
        comparison_df = compute_metrics(trained_models, y_test, problem_type)
        set_state("model_comparison", comparison_df)

    st.markdown(
        """
        Compare the prediction quality and execution speeds of all trained models.
        Review feature importances below to see which columns drove model predictions.
        """
    )

    # ── Comparison Table ──────────────────────────────────────────────
    st.subheader("Model Evaluation Summary")
    st.dataframe(comparison_df.round(4), use_container_width=True)

    st.divider()

    # ── Performance Charts ─────────────────────────────────────────────
    st.subheader("Visual Comparisons")
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        # Compare main accuracy/score
        main_metric = "Accuracy" if problem_type == "Classification" else "R² Score"
        fig_score = px.bar(
            comparison_df,
            x="Model Name",
            y=main_metric,
            title=f"Model Comparison: {main_metric}",
            color="Model Name"
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with col_plot2:
        # Compare training speeds
        fig_time = px.bar(
            comparison_df,
            x="Model Name",
            y="Fit Time (s)",
            title="Model Training Latency (lower is better)",
            color="Model Name"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # ── Feature Importance Tab ─────────────────────────────────────────
    st.subheader("Feature Importances")
    
    # Model selector
    selected_model_name = st.selectbox(
        "Select a model to view feature importances:",
        options=list(trained_models.keys())
    )

    if selected_model_name:
        model_info = trained_models[selected_model_name]
        model_instance = model_info["instance"]
        
        imp_df = extract_feature_importance(model_instance, feature_names)
        if imp_df is not None:
            fig_imp = px.bar(
                imp_df.head(10).sort_values(by="Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top 10 Feature Importances for {selected_model_name}",
                labels={"Importance": "Score/Weight", "Feature": "Feature Name"}
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(
                f"Feature importance is not directly supported by **{selected_model_name}** "
                "(e.g., custom algorithms or distance-based estimators like KNN/SVM from scratch). "
                "You can inspect SHAP explainability for this model on the **Explainable AI** page."
            )


# Add helper type to import
render_model_comparison_page()
