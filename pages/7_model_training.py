"""SmartML Studio — Model Training page."""

import logging
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from config.logging_config import setup_logging
from src.models.model_trainer import ModelTrainer
from utils.session_state import get_state, set_state, reset_downstream
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_model_training_page() -> None:
    """Render the Model Training page."""
    st.header("Model Training")

    # Prerequisite check: check if we have preprocessed data and selected models
    profile = get_state("profile")
    df = get_state("feature_engineered_data")
    if df is None:
        df = get_state("preprocessed_data")

    selected_models = get_state("model_recommendations")

    if df is None or profile is None:
        st.warning("Please complete Dataset Analysis first.")
        st.stop()

    if not selected_models:
        st.warning("Please select models first in the Smart Model Advisor page.")
        st.stop()

    target_col = get_state("target_column")
    problem_type = get_state("problem_type")

    st.markdown(
        f"You have selected the following models to train: **{', '.join(selected_models)}**. "
        "Adjust the Train/Test split ratio and click **Train Models** below."
    )

    # ── Parameters ────────────────────────────────────────────────────
    st.subheader("Training Parameters")
    test_size_pct = st.slider("Test Dataset Split Ratio (%):", min_value=10, max_value=50, value=20, step=5)
    test_size = test_size_pct / 100.0

    st.divider()

    train_btn = st.button("Train Models", type="primary")

    if train_btn:
        # Separate X and y
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
        feature_names = list(df.drop(columns=[target_col]).columns)


        # Handle splitting
        # If classification, apply stratify if target has >= 2 classes and all classes have >= 2 samples
        class_counts = profile.get("class_balance", {})
        has_small_classes = any(count < 2 for count in class_counts.values())
        unique_classes = len(class_counts)
        
        stratify = y if (problem_type == "Classification" and unique_classes >= 2 and not has_small_classes) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )


        # Store split data in state for downstream modules
        set_state("X_train", X_train)
        set_state("X_test", X_test)
        set_state("y_train", y_train)
        set_state("y_test", y_test)
        set_state("feature_names", feature_names)  # For feature importance and SHAP

        # Fit models
        trainer = ModelTrainer(problem_type=problem_type)
        trained_results = {}

        progress_text = "Training in progress. Please wait..."
        progress_bar = st.progress(0, text=progress_text)
        
        try:
            for idx, model_name in enumerate(selected_models):
                progress_val = int(((idx) / len(selected_models)) * 100)
                progress_bar.progress(progress_val, text=f"Training: {model_name}...")
                
                # Single model train
                single_res = trainer.train_models(X_train, y_train, X_test, y_test, [model_name])
                trained_results.update(single_res)

            progress_bar.progress(100, text="Training completed successfully!")
            
            # Save to state
            set_state("trained_models", trained_results)
            reset_downstream("y_test")


            st.success("All selected models have been trained successfully!")

            # Quick recap table
            recap_data = []
            for name, res in trained_results.items():
                recap_data.append({
                    "Model Name": name,
                    "Fit Time (seconds)": f"{res['fit_time']:.4f}",
                    "Prediction Time (seconds)": f"{res['predict_time']:.4f}"
                })
            st.dataframe(pd.DataFrame(recap_data), use_container_width=True)

            st.info("You can now proceed to the **Model Comparison** page in the sidebar.")

        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            logger.error("Failed model training phase: %s", str(e), exc_info=True)

    elif get_state("trained_models") is not None:
        st.success("Previously trained models exist in session state.")
        recap_data = []
        for name, res in get_state("trained_models").items():
            recap_data.append({
                "Model Name": name,
                "Fit Time (seconds)": f"{res['fit_time']:.4f}",
                "Prediction Time (seconds)": f"{res['predict_time']:.4f}"
            })
        st.dataframe(pd.DataFrame(recap_data), use_container_width=True)



render_model_training_page()
