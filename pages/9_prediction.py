import logging
import pandas as pd
import numpy as np
import streamlit as st

from config.logging_config import setup_logging
from src.ensemble.hybrid_ensemble import HybridEnsemble
from utils.session_state import get_state, set_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_prediction_page() -> None:
    """Render the Prediction page."""
    st.header("Prediction")

    # Prerequisite guards
    trained_models = get_state("trained_models")
    X_test = get_state("X_test")
    y_test = get_state("y_test")
    problem_type = get_state("problem_type")
    target_column = get_state("target_column")
    feature_names = get_state("feature_names")

    if trained_models is None or X_test is None or y_test is None:
        st.warning("Please train models first in the Model Training page.")
        st.stop()

    st.markdown(
        """
        Generate predictions using either a single trained model or a custom **Hybrid Ensemble**
        combining multiple models with voting (classification) or averaging (regression).
        """
    )

    # Model Selection Mode
    pred_mode = st.radio(
        "Select Prediction Mode:",
        options=["Single Model", "Custom Hybrid Ensemble"],
        index=0
    )

    selected_model_name = None
    ensemble_voting = None
    ensemble_weights = None
    selected_ensemble_models = []

    if pred_mode == "Single Model":
        st.subheader("Configure Single Model")
        selected_model_name = st.selectbox(
            "Select Model:",
            options=list(trained_models.keys())
        )
    else:
        st.subheader("Configure Custom Hybrid Ensemble")
        selected_ensemble_models = st.multiselect(
            "Select Models to Include in the Ensemble:",
            options=list(trained_models.keys()),
            default=list(trained_models.keys())
        )

        if len(selected_ensemble_models) < 1:
            st.warning("Please select at least one model to build the ensemble.")
        else:
            voting_options = (
                ["majority", "weighted"]
                if problem_type == "Classification"
                else ["average", "weighted"]
            )
            ensemble_voting = st.selectbox(
                "Voting / Aggregation Strategy:",
                options=voting_options,
                format_func=lambda x: x.title()
            )

            if ensemble_voting == "weighted":
                st.markdown("**Assign Weights for Selected Models (weights will be normalized):**")
                ensemble_weights = []
                for model_name in selected_ensemble_models:
                    w = st.slider(f"Weight for {model_name}:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    ensemble_weights.append(w)

    st.divider()

    generate_btn = st.button("Generate Predictions", type="primary")

    if generate_btn:
        if pred_mode == "Custom Hybrid Ensemble" and len(selected_ensemble_models) < 1:
            st.error("Cannot generate predictions: Ensemble must have at least one model.")
            st.stop()

        try:
            with st.spinner("Generating predictions..."):
                if pred_mode == "Single Model":
                    # Retrieve the fitted instance
                    model_dict = trained_models[selected_model_name]
                    model_instance = model_dict["instance"]
                    y_pred = model_instance.predict(X_test)
                    
                    # Store selected model state
                    set_state("selected_prediction_model", selected_model_name)
                    set_state("ensemble", None)
                else:
                    # Retrieve model instances
                    base_models = [trained_models[m]["instance"] for m in selected_ensemble_models]
                    
                    # Instantiate and fit ensemble
                    ensemble = HybridEnsemble(
                        voting=ensemble_voting,
                        weights=ensemble_weights,
                        problem_type=problem_type
                    )
                    ensemble.fit(base_models)
                    y_pred = ensemble.predict(X_test)

                    # Store selected model state
                    set_state("selected_prediction_model", "ensemble")
                    set_state("ensemble", ensemble)

                # Format predictions dataframe for preview
                preview_df = pd.DataFrame(X_test, columns=feature_names)
                
                # Check if label encoder exists to display actual category labels instead of integer encodings
                le = get_state("label_encoder")
                if le is not None:
                    try:
                        preview_df[f"{target_column} (Actual)"] = le.inverse_transform(y_test.astype(int))
                        preview_df[f"{target_column} (Predicted)"] = le.inverse_transform(y_pred.astype(int))
                    except Exception:
                        preview_df[f"{target_column} (Actual)"] = y_test
                        preview_df[f"{target_column} (Predicted)"] = y_pred
                else:
                    preview_df[f"{target_column} (Actual)"] = y_test
                    preview_df[f"{target_column} (Predicted)"] = y_pred


                # Store predictions
                set_state("predictions", preview_df)

                # Update evaluation report
                report = get_state("evaluation_report") or {}
                report["prediction_mode"] = pred_mode
                report["selected_model"] = selected_model_name if pred_mode == "Single Model" else "Custom Hybrid Ensemble"
                report["ensemble_voting"] = ensemble_voting
                report["ensemble_models"] = selected_ensemble_models
                set_state("evaluation_report", report)

                st.success("Predictions generated successfully!")

        except Exception as e:
            st.error(f"Failed to generate predictions: {str(e)}")
            logger.error("Prediction generation failed: %s", str(e), exc_info=True)


    # ── Prediction Results Preview ───────────────────────────────────────
    predictions_df = get_state("predictions")
    if predictions_df is not None:
        st.subheader("Prediction Results Preview (Test Set)")
        st.dataframe(predictions_df.round(4), use_container_width=True)


render_prediction_page()

