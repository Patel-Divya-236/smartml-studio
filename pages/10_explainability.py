import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config.logging_config import setup_logging
from src.explainability.explainer import ModelExplainer
from utils.session_state import get_state, set_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_explainable_ai_page() -> None:
    """Render the Explainable AI page."""
    st.header("Explainable AI")

    # Prerequisite guards
    trained_models = get_state("trained_models")
    X_train = get_state("X_train")
    X_test = get_state("X_test")
    y_test = get_state("y_test")
    problem_type = get_state("problem_type")
    feature_names = get_state("feature_names")

    if trained_models is None or X_train is None or X_test is None or y_test is None:
        st.warning("Please train models first in the Model Training page.")
        st.stop()

    st.markdown(
        """
        Explore how trained models make decisions. This page provides global explanations (what features
        drive the model overall) and local explanations (why a specific prediction was made) using **SHAP values**,
        accompanied by diagnostic performance curves.
        """
    )

    # Model Selector
    selected_model_name = st.selectbox(
        "Choose a model to explain:",
        options=list(trained_models.keys())
    )

    if not selected_model_name:
        st.stop()

    model_dict = trained_models[selected_model_name]
    model_instance = model_dict["instance"]
    y_pred = model_dict["y_pred"]
    y_prob = model_dict["y_prob"]

    # Compute or load SHAP values
    st.subheader("1. SHAP Model Explanations (Global & Local)")
    
    # Initialize explainer
    explainer = ModelExplainer()
    
    with st.spinner("Computing SHAP values... (Kernel SHAP fallback is restricted to subsets to keep it fast)"):
        try:
            explanation = explainer.explain(model_instance, X_train, X_test, feature_names)
            shap_vals = explanation["shap_values"]
            explainer_type = explanation["explainer_type"]
            is_subset = explanation["is_subset"]
            
            st.info(f"Calculated SHAP values using **{explainer_type}** Explainer.")
            if is_subset:
                st.warning("To keep execution times fast, SHAP fallback evaluated only the first 10 samples.")

            # Tab layout for Global vs Local
            tab_global, tab_local = st.tabs(["Global Interpretability", "Local Explanations"])

            with tab_global:
                st.markdown("#### Global Feature Impact")
                st.write("This plot shows the general importance of features across the test set.")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                import shap
                
                # Check structure of shap_vals
                if isinstance(shap_vals, list):
                    # Multiclass classification: select class 0 for representation
                    shap.summary_plot(shap_vals[0], X_test[:len(shap_vals[0])] if is_subset else X_test, feature_names=feature_names, show=False)
                elif hasattr(shap_vals, "values"):
                    shap.summary_plot(shap_vals.values, X_test[:len(shap_vals.values)] if is_subset else X_test, feature_names=feature_names, show=False)
                else:
                    shap.summary_plot(shap_vals, X_test[:len(shap_vals)] if is_subset else X_test, feature_names=feature_names, show=False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with tab_local:
                st.markdown("#### Explain a Single Prediction")
                max_idx = 9 if is_subset else len(X_test) - 1
                sample_idx = st.slider("Select Test Sample Index:", min_value=0, max_value=max_idx, value=0)
                
                le = get_state("label_encoder")
                if le is not None:
                    actual_label = le.inverse_transform([y_test[sample_idx]])[0]
                    pred_label = le.inverse_transform([y_pred[sample_idx]])[0]
                else:
                    actual_label = y_test[sample_idx]
                    pred_label = y_pred[sample_idx]
                
                st.markdown(f"**Actual Value:** `{actual_label}` | **Predicted Value:** `{pred_label}`")

                # Local Waterfall or Bar chart representation
                fig_local, ax_local = plt.subplots(figsize=(8, 4))
                
                try:
                    # Attempt waterfall plot
                    if hasattr(shap_vals, "values") and not isinstance(shap_vals, list):
                        shap.plots.waterfall(shap_vals[sample_idx], show=False)
                    else:
                        # Fallback custom bar plot for local features
                        if isinstance(shap_vals, list):
                            vals = shap_vals[0][sample_idx]
                        else:
                            vals = shap_vals[sample_idx]
                        
                        df_local = pd.DataFrame({
                            "Feature": feature_names,
                            "SHAP Value": vals
                        }).sort_values(by="SHAP Value", key=abs, ascending=True)

                        colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in df_local["SHAP Value"]]
                        plt.barh(df_local["Feature"], df_local["SHAP Value"], color=colors)
                        plt.axvline(0, color="gray", linestyle="--")
                        plt.title("Feature Contributions (Positive = drives prediction higher)")
                    
                    plt.tight_layout()
                    st.pyplot(fig_local)
                except Exception as inner_err:
                    st.error(f"Could not render local plot: {str(inner_err)}")
                finally:
                    plt.close(fig_local)

        except Exception as e:
            st.error(f"Could not compute SHAP explanations: {str(e)}")
            st.info("Showing standard feature importances fallback in the **Model Comparison** tab.")


    st.divider()

    # Diagnostic Curves / Plotly Charts
    st.subheader("2. Model Evaluation Diagnostics")
    
    if problem_type == "Classification":
        col_curve1, col_curve2 = st.columns(2)

        with col_curve1:
            st.markdown("#### Confusion Matrix")
            # Force confusion matrix to span all potential labels to avoid dimension mismatches in Plotly
            le = get_state("label_encoder")
            if le is not None:
                labels = list(le.classes_)
                cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
            else:
                labels = list(np.unique(y_test))
                cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=labels,
                y=labels,
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                color_continuous_scale="Blues",
                title=f"Confusion Matrix for {selected_model_name}"
            )

            st.plotly_chart(fig_cm, use_container_width=True)


        with col_curve2:
            st.markdown("#### ROC Curve")
            if y_prob is not None:
                try:
                    # Handle binary vs multiclass
                    if len(np.unique(y_test)) == 2:
                        prob_pos = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
                        fpr, tpr, _ = roc_curve(y_test, prob_pos)
                        roc_auc = auc(fpr, tpr)

                        fig_roc = px.line(
                            x=fpr, y=tpr,
                            labels=dict(x="False Positive Rate", y="True Positive Rate"),
                            title=f"ROC Curve (AUC = {roc_auc:.4f})"
                        )
                        fig_roc.add_shape(
                            type="line", line=dict(dash="dash"),
                            x0=0, x1=1, y0=0, y1=1
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC curves for multi-class targets are skipped here to keep the visualization clean. Check AUC scores in Model Comparison.")
                except Exception as e:
                    st.info(f"Could not construct ROC curve: {str(e)}")
            else:
                st.info("Probabilities/Decision Scores not available for this model to compute ROC Curve.")

    else:
        # Regression diagnostic curves
        col_curve1, col_curve2 = st.columns(2)

        with col_curve1:
            st.markdown("#### Actual vs Predicted Scatter")
            fig_ap = px.scatter(
                x=y_test, y=y_pred,
                labels=dict(x="Actual Values", y="Predicted Values"),
                title=f"Actual vs Predicted for {selected_model_name}"
            )
            # Add y=x line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_ap.add_shape(
                type="line", line=dict(color="red", dash="dash"),
                x0=min_val, x1=max_val, y0=min_val, y1=max_val
            )
            st.plotly_chart(fig_ap, use_container_width=True)

        with col_curve2:
            st.markdown("#### Residuals Plot")
            residuals = y_test - y_pred
            fig_res = px.scatter(
                x=y_pred, y=residuals,
                labels=dict(x="Predicted Values", y="Residuals"),
                title="Residuals (Errors) Plot"
            )
            fig_res.add_shape(
                type="line", line=dict(color="red", dash="dash"),
                x0=y_pred.min(), x1=y_pred.max(), y0=0, y1=0
            )
            st.plotly_chart(fig_res, use_container_width=True)


render_explainable_ai_page()

