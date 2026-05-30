"""SmartML Studio — Feature Engineering page."""

import logging
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from config.logging_config import setup_logging
from config.settings import SETTINGS
from utils.session_state import get_state, set_state, reset_downstream
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def apply_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    config: dict
) -> pd.DataFrame:
    """Apply feature engineering steps sequentially based on configuration."""
    logger.info("Applying feature engineering steps...")
    
    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # ── 1. Low Variance Filter ──────────────────────────────────────
    if config.get("low_variance_active", False):
        threshold = config.get("low_variance_threshold", 0.01)
        logger.info("Applying Variance Threshold with threshold=%s", threshold)
        
        # Scikit-learn VarianceThreshold drops numeric columns with low variance
        # Make sure we only apply to numeric columns to avoid error on object columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X[numeric_cols])
            
            # Identify columns to drop
            kept_features = numeric_cols[selector.get_support()]
            dropped_features = list(set(numeric_cols) - set(kept_features))
            
            if dropped_features:
                X = X.drop(columns=dropped_features)
                logger.info("Dropped low-variance columns: %s", dropped_features)

    # ── 2. Polynomial Features ──────────────────────────────────────
    if config.get("poly_active", False):
        degree = config.get("poly_degree", 2)
        interaction_only = config.get("poly_interaction_only", False)
        logger.info("Applying Polynomial Features with degree=%d", degree)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
            poly_features = poly.fit_transform(X[numeric_cols])
            poly_cols = poly.get_feature_names_out(numeric_cols)
            
            # Combine back with non-numeric columns
            non_numeric_cols = list(set(X.columns) - set(numeric_cols))
            X_poly = pd.DataFrame(poly_features, columns=poly_cols, index=X.index)
            if non_numeric_cols:
                X = pd.concat([X_poly, X[non_numeric_cols]], axis=1)
            else:
                X = X_poly

    # ── 3. Principal Component Analysis (PCA) ───────────────────────
    if config.get("pca_active", False):
        n_components = config.get("pca_components", 2)
        logger.info("Applying PCA with n_components=%d", n_components)
        
        # PCA only works on numeric data
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Clip components to count of numeric columns
            n_components = min(n_components, len(numeric_cols))
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(X[numeric_cols])
            pca_cols = [f"PC{i+1}" for i in range(n_components)]
            
            # Combine back with non-numeric
            non_numeric_cols = list(set(X.columns) - set(numeric_cols))
            X_pca = pd.DataFrame(pca_features, columns=pca_cols, index=X.index)
            if non_numeric_cols:
                X = pd.concat([X_pca, X[non_numeric_cols]], axis=1)
            else:
                X = X_pca

    # ── 4. Feature Selection (SelectKBest) ──────────────────────────
    if config.get("select_k_best_active", False):
        k = config.get("select_k_best_k", 5)
        method_name = config.get("select_k_best_method", "ANOVA")
        logger.info("Applying SelectKBest with k=%d, method=%s", k, method_name)
        
        # Determine score function
        if problem_type == "Classification":
            score_func = f_classif if method_name == "ANOVA" else mutual_info_classif
        else:
            score_func = f_regression if method_name == "ANOVA" else mutual_info_regression
            
        # Ensure k is not larger than feature count
        k = min(k, X.shape[1])
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        selected_cols = X.columns[selector.get_support()]
        X = X[selected_cols]
        logger.info("Selected top %d features: %s", k, list(selected_cols))

    # Recombine features and target
    engineered_df = pd.concat([X, y], axis=1)
    logger.info("Feature engineering complete. Shape: %s", engineered_df.shape)
    return engineered_df


def render_feature_engineering_page() -> None:
    """Render the Feature Engineering page."""
    st.header("Feature Engineering")

    preprocessed_df = get_state("preprocessed_data")
    target_col = get_state("target_column")
    problem_type = get_state("problem_type")

    # Prerequisite guard
    if preprocessed_df is None:
        st.warning("Please complete Preprocessing first.")
        st.stop()

    st.markdown(
        """
        Enhance your features using optional techniques like Dimensionality Reduction (PCA),
        Feature Selection, or Polynomial Interactions. Configure the choices below, or skip this
        module to train models on the basic preprocessed features.
        """
    )

    rows, cols = preprocessed_df.shape
    st.write(f"Current Preprocessed Features Shape: **{rows} rows, {cols - 1} features** (excluding target)")

    st.subheader("Select and Configure Techniques")
    
    # Configure low variance removal
    low_var_active = st.checkbox("Apply Low-Variance Filter", value=False)
    low_var_thresh = 0.01
    if low_var_active:
        low_var_thresh = st.slider("Minimum Variance Threshold:", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

    st.divider()

    # Configure polynomial features
    poly_active = st.checkbox("Apply Polynomial / Interaction Features", value=False)
    poly_degree = 2
    poly_interact = False
    if poly_active:
        col_poly1, col_poly2 = st.columns(2)
        with col_poly1:
            poly_degree = st.selectbox("Polynomial Degree:", options=[2, 3], index=0)
        with col_poly2:
            poly_interact = st.checkbox("Interaction Only (no x² term)", value=False)

    st.divider()

    # Configure PCA
    pca_active = st.checkbox("Apply PCA (Principal Component Analysis)", value=False)
    pca_comps = 2
    if pca_active:
        max_comps = max(1, preprocessed_df.shape[1] - 1)
        pca_comps = st.slider("Number of Principal Components:", min_value=1, max_value=max_comps, value=min(2, max_comps))

    st.divider()

    # Configure Feature Selection
    sel_active = st.checkbox("Apply Feature Selection (SelectKBest)", value=False)
    sel_k = 5
    sel_method = "ANOVA"
    if sel_active:
        max_k = max(1, preprocessed_df.shape[1] - 1)
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel_k = st.slider("Number of top features to keep (k):", min_value=1, max_value=max_k, value=min(5, max_k))
        with col_sel2:
            sel_method = st.selectbox("Score Function:", options=["ANOVA (F-value)", "Mutual Information"])

    st.divider()

    # Actions panel
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        apply_btn = st.button("Apply Feature Engineering", type="primary")
    with col_act2:
        skip_btn = st.button("Skip & Use Preprocessed Features")

    if apply_btn:
        config = {
            "low_variance_active": low_var_active,
            "low_variance_threshold": low_var_thresh,
            "poly_active": poly_active,
            "poly_degree": poly_degree,
            "poly_interaction_only": poly_interact,
            "pca_active": pca_active,
            "pca_components": pca_comps,
            "select_k_best_active": sel_active,
            "select_k_best_k": sel_k,
            "select_k_best_method": "ANOVA" if "ANOVA" in sel_method else "MutualInfo"
        }

        with st.spinner("Engineering features..."):
            try:
                engineered_df = apply_feature_engineering(preprocessed_df, target_col, problem_type, config)
                set_state("feature_engineered_data", engineered_df)
                reset_downstream("feature_engineered_data")
                st.success("Feature Engineering successfully applied!")
                st.dataframe(engineered_df.head(5), use_container_width=True)
                st.info("You can now proceed to the **Smart Model Advisor** page in the sidebar.")
            except Exception as e:
                st.error(f"Error applying feature engineering: {str(e)}")
                logger.error("Failed to run feature engineering: %s", str(e), exc_info=True)

    elif skip_btn:
        logger.info("Skipping feature engineering, using preprocessed data directly.")
        set_state("feature_engineered_data", preprocessed_df)
        reset_downstream("feature_engineered_data")
        st.info("Skipped. Using the default preprocessed features.")
        st.dataframe(preprocessed_df.head(5), use_container_width=True)
        st.info("You can now proceed to the **Smart Model Advisor** page in the sidebar.")

    elif get_state("feature_engineered_data") is not None:
        st.success("Feature engineered data exists in session state.")
        st.dataframe(get_state("feature_engineered_data").head(5), use_container_width=True)



render_feature_engineering_page()
