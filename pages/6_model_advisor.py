"""SmartML Studio — Smart Model Advisor page."""

import logging
import streamlit as st

from config.logging_config import setup_logging
from src.advisors.model_advisor import ModelAdvisor
from utils.session_state import get_state, set_state, reset_downstream
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_model_advisor_page() -> None:
    """Render the Smart Model Advisor page."""
    st.header("Smart Model Advisor")

    profile = get_state("profile")
    df = get_state("feature_engineered_data")
    if df is None:
        # Fallback to preprocessed data
        df = get_state("preprocessed_data")

    # Prerequisite guard
    if df is None or profile is None:
        st.warning("Please complete Dataset Analysis first.")
        st.stop()

    st.markdown(
        """
        The Model Advisor recommends classification or regression algorithms
        based on target type, row count, dimension count, and class balances.
        Select the models you wish to train in the next module.
        """
    )

    rows, cols = df.shape
    st.write(f"Training Features Shape: **{rows} rows, {cols - 1} features** (excluding target)")

    # Run advisor
    advisor = ModelAdvisor()
    recs = advisor.recommend(profile)

    # ── Selection checklist UI ────────────────────────────────────────
    st.subheader("Recommended Models")
    
    selected_models = []

    # Restore previous selections if any
    prev_selected = get_state("model_recommendations") or []

    for i, rec in enumerate(recs):
        model_name = rec.metadata.get("model_name")
        if not model_name:
            continue
            
        col_chk, col_why = st.columns([0.7, 0.3])
        with col_chk:
            label_text = f"**{rec.label}** (Rating: {rec.star_rating}/5 | {rec.confidence_score * 100:.0f}% confidence) — *{rec.reason}*"
            
            # Default checked if it was previously checked, or if there is no previous selection (default True)
            default_val = model_name in prev_selected if prev_selected else True
            
            is_selected = st.checkbox(label_text, value=default_val, key=f"chk_model_{i}")
            if is_selected:
                selected_models.append(model_name)
        with col_why:
            with st.expander("Why?"):
                st.write(rec.why_explanation)

    st.divider()

    confirm_btn = st.button("Confirm Model Selection", type="primary")

    if confirm_btn:
        if not selected_models:
            st.error("Please select at least one model to train.")
        else:
            set_state("model_recommendations", selected_models)
            reset_downstream("model_recommendations")
            st.success(f"Selected models: **{', '.join(selected_models)}**. Proceed to the **Model Training** page in the sidebar.")
            
    elif get_state("model_recommendations") is not None:
        st.info(f"Previously selected models: **{', '.join(get_state('model_recommendations'))}**")



render_model_advisor_page()
