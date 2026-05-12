"""SmartML Studio — Intelligent End-to-End ML Platform.

Entry point for the Streamlit application. Configures the page,
initialises logging and session state, and renders the landing page.
"""

import logging

import streamlit as st

from config.logging_config import setup_logging
from utils.session_state import init_session_state
from utils.styling import apply_custom_theme

# ── Bootstrap (runs once per session) ──────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SmartML Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
apply_custom_theme()


# ── Sidebar branding ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin: 0; font-size: 1.6rem;">SmartML Studio</h1>
            <p style="margin: 0.25rem 0 0; font-size: 0.85rem; opacity: 0.7;">
                Intelligent ML Platform
            </p>
        </div>
        <hr style="margin: 0.5rem 0;">
        """,
        unsafe_allow_html=True,
    )
    st.caption("Navigate using the pages above")

# ── Landing page ──────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.25rem;">
            SmartML Studio
        </h1>
        <p style="font-size: 1.15rem; opacity: 0.8; max-width: 640px; margin: auto;">
            An <strong>Intelligent End-to-End Machine Learning Platform</strong> for
            structured tabular data — with confidence-scored recommendations at every step.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Module overview cards ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Upload & Analyse")
    st.markdown(
        "Upload any CSV/Excel dataset, select a target column, "
        "and get a comprehensive statistical profile instantly."
    )
    st.markdown("### Smart Preprocessing")
    st.markdown(
        "Receive per-column imputation, encoding, and scaling "
        "recommendations — each with a confidence score and reason."
    )
    st.markdown("### Compare Models")
    st.markdown(
        "Side-by-side comparison of accuracy, precision, recall, "
        "F1, ROC-AUC, training time, and feature importance."
    )

with col2:
    st.markdown("### Smart Visualizations")
    st.markdown(
        "Get ranked chart recommendations based on your data's "
        "characteristics. Accept, remove, or reorder — then generate."
    )
    st.markdown("### Feature Engineering")
    st.markdown(
        "Opt-in PCA, polynomial features, feature selection, and "
        "low-variance removal — all user-triggered."
    )
    st.markdown("### Predict")
    st.markdown(
        "Use a single model or a custom hybrid ensemble "
        "(majority/weighted voting) on new data."
    )

with col3:
    st.markdown("### Model Advisor")
    st.markdown(
        "Smart model recommendations based on dataset size, "
        "feature types, class balance, and complexity."
    )
    st.markdown("### Train Models")
    st.markdown(
        "Train sklearn models plus from-scratch SVM and kNN. "
        "Multi-select which models to train."
    )
    st.markdown("### Explainable AI")
    st.markdown(
        "SHAP summary plots, per-prediction explanations, "
        "confusion matrices, ROC curves, and learning curves."
    )

st.divider()

st.markdown(
    """
    <div style="text-align: center; opacity: 0.6; font-size: 0.85rem;">
        Use the sidebar to navigate through each module.
        Start with Dataset Upload.
    </div>
    """,
    unsafe_allow_html=True,
)


logger.info("Landing page rendered.")
