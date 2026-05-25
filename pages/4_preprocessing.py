"""SmartML Studio — Smart Preprocessing page."""

import logging
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

from config.logging_config import setup_logging
from src.advisors.preprocessing_advisor import PreprocessingAdvisor
from utils.session_state import get_state, set_state, reset_downstream
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def apply_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    impute_config: dict,
    encode_config: dict,
    scale_config: dict
) -> pd.DataFrame:
    """Execute preprocessing strategies on the dataframe based on configs."""
    logger.info("Applying preprocessing pipeline...")
    
    # Work on a copy of dataframe to prevent modifying original cached data
    working_df = df.copy()

    # ── 1. Drop Configured Columns ───────────────────────────────
    cols_to_drop = [col for col, action in impute_config.items() if action == "Drop Column"]
    # Never drop target
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    
    if cols_to_drop:
        working_df = working_df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    # Re-identify active columns (excluding target column)
    feature_cols = [c for c in working_df.columns if c != target_col]

    # ── 2. Imputation ────────────────────────────────────────────
    for col in feature_cols:
        action = impute_config.get(col, "None")
        if action == "None" or working_df[col].isnull().sum() == 0:
            continue
        
        logger.info("Imputing column %s using %s", col, action)
        if action == "Mean":
            imputer = SimpleImputer(strategy="mean")
            working_df[[col]] = imputer.fit_transform(working_df[[col]])
        elif action == "Median":
            imputer = SimpleImputer(strategy="median")
            working_df[[col]] = imputer.fit_transform(working_df[[col]])
        elif action == "Mode" or action == "Most Frequent":
            imputer = SimpleImputer(strategy="most_frequent")
            working_df[[col]] = imputer.fit_transform(working_df[[col]])
        elif action == "KNN":
            imputer = KNNImputer(n_neighbors=5)
            working_df[[col]] = imputer.fit_transform(working_df[[col]])

    # Separate target
    y = working_df[target_col].copy()
    
    # Label encode target if classification to support all ML algorithms (like XGBoost)
    problem_type = get_state("problem_type")
    if problem_type == "Classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
        set_state("label_encoder", le)
    else:
        set_state("label_encoder", None)

    X = working_df.drop(columns=[target_col])


    # ── 3. Encoding & Scaling ─────────────────────────────────────
    processed_parts = []
    
    for col in X.columns:
        col_series = X[[col]].copy()
        
        # Scaling / transformation
        scale_act = scale_config.get(col, "None")
        if scale_act == "Log1p":
            col_series = np.log1p(col_series)
            # Apply normal scaler after log transform if needed, default standard
            scaler = StandardScaler()
            col_series = pd.DataFrame(scaler.fit_transform(col_series), columns=[col], index=col_series.index)
        elif scale_act == "Standard":
            scaler = StandardScaler()
            col_series = pd.DataFrame(scaler.fit_transform(col_series), columns=[col], index=col_series.index)
        elif scale_act == "MinMax":
            scaler = MinMaxScaler()
            col_series = pd.DataFrame(scaler.fit_transform(col_series), columns=[col], index=col_series.index)
        elif scale_act == "Robust":
            scaler = RobustScaler()
            col_series = pd.DataFrame(scaler.fit_transform(col_series), columns=[col], index=col_series.index)

        # Encoding
        encode_act = encode_config.get(col, "None")
        if encode_act == "One-Hot":
            encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
            encoded_arr = encoder.fit_transform(col_series.astype(str))
            cat_names = encoder.get_feature_names_out([col])
            col_df = pd.DataFrame(encoded_arr, columns=cat_names, index=col_series.index)
        elif encode_act == "Ordinal/Label":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoded_arr = encoder.fit_transform(col_series.astype(str))
            col_df = pd.DataFrame(encoded_arr, columns=[col], index=col_series.index)
        else:
            col_df = col_series

        processed_parts.append(col_df)

    # Reassemble DataFrame
    if processed_parts:
        X_processed = pd.concat(processed_parts, axis=1)
    else:
        X_processed = pd.DataFrame(index=working_df.index)

    # Add back target
    # If target has missing, drop those rows
    if y.isnull().sum() > 0:
        logger.info("Dropping rows with missing target values")
        valid_indices = y.dropna().index
        X_processed = X_processed.loc[valid_indices]
        y = y.loc[valid_indices]

    preprocessed_df = pd.concat([X_processed, y], axis=1)
    logger.info("Preprocessing complete. Shape: %s", preprocessed_df.shape)
    return preprocessed_df


def render_preprocessing_page() -> None:
    """Render the Smart Preprocessing page."""
    st.header("Smart Preprocessing")

    df = get_state("dataset")
    profile = get_state("profile")
    target_col = get_state("target_column")

    # Prerequisite guard
    if df is None or profile is None:
        st.warning("Please complete Dataset Analysis first.")
        st.stop()

    recs = get_state("preprocessing_recommendations")
    if recs is None:
        advisor = PreprocessingAdvisor()
        recs = advisor.recommend(profile)
        set_state("preprocessing_recommendations", recs)

    st.markdown(
        """
        The Preprocessing Advisor recommends imputation, encoding, and scaling
        strategies for each feature. Review these per-column plans, modify them if needed,
        and click **Apply Preprocessing** to generate the cleaned dataset.
        """
    )

    # Helper maps for defaults based on advisors recommendations
    default_impute = {}
    default_encode = {}
    default_scale = {}

    for rec in recs:
        col = rec.metadata.get("column")
        act = rec.metadata.get("action")
        cat = rec.category
        if col:
            if cat == "imputation":
                if act == "drop":
                    default_impute[col] = "Drop Column"
                elif act == "median":
                    default_impute[col] = "Median"
                elif act == "knn":
                    default_impute[col] = "KNN"
                elif act == "mode":
                    default_impute[col] = "Mode"
            elif cat == "encoding":
                if act == "onehot":
                    default_encode[col] = "One-Hot"
                elif act == "ordinal":
                    default_encode[col] = "Ordinal/Label"
            elif cat == "scaling":
                if act == "log1p":
                    default_scale[col] = "Log1p"
                elif act == "robust":
                    default_scale[col] = "Robust"
                elif act == "standard":
                    default_scale[col] = "Standard"

    # Per-column config UI
    impute_selections = {}
    encode_selections = {}
    scale_selections = {}

    st.subheader("Configure Pipeline Settings")
    for col in df.columns:
        if col == target_col:
            continue
        
        # Display column name, type, details
        is_num = col in profile["numeric_columns"]
        col_type = "Numeric" if is_num else "Categorical"
        missing_cnt = profile["missing_values"].get(col, 0)
        missing_p = profile["missing_pct"].get(col, 0.0)
        card = profile["cardinality"].get(col, 0)
        
        with st.expander(f"Column: **{col}** ({col_type}) — Missing: {missing_p:.1f}% | Card: {card}"):
            # Imputer selector
            impute_opts = ["None", "Mean", "Median", "Mode", "KNN", "Drop Column"]
            # Set default
            def_imp = default_impute.get(col, "None")
            if def_imp not in impute_opts:
                def_imp = "None"
            
            # Encoder selector
            encode_opts = ["None", "One-Hot", "Ordinal/Label"]
            def_enc = "None"
            if not is_num:
                def_enc = default_encode.get(col, "One-Hot")

            # Scaler selector
            scale_opts = ["None", "Standard", "MinMax", "Robust", "Log1p"]
            def_scl = "None"
            if is_num:
                def_scl = default_scale.get(col, "Standard")

            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                impute_selections[col] = st.selectbox(
                    f"Imputer for {col}:",
                    options=impute_opts,
                    index=impute_opts.index(def_imp),
                    key=f"imp_sel_{col}"
                )
            with col_sel2:
                encode_selections[col] = st.selectbox(
                    f"Encoder for {col}:",
                    options=encode_opts,
                    index=encode_opts.index(def_enc),
                    key=f"enc_sel_{col}"
                )
            with col_sel3:
                scale_selections[col] = st.selectbox(
                    f"Scaler for {col}:",
                    options=scale_opts,
                    index=scale_opts.index(def_scl),
                    key=f"scl_sel_{col}"
                )

    st.divider()

    # Trigger preprocessing execution
    apply_btn = st.button("Apply Preprocessing", type="primary")

    if apply_btn:
        with st.spinner("Processing column transformations..."):
            try:
                preprocessed_df = apply_preprocessing_pipeline(
                    df,
                    target_col,
                    impute_selections,
                    encode_selections,
                    scale_selections
                )
                set_state("preprocessed_data", preprocessed_df)
                reset_downstream("preprocessed_data")
                st.success("Preprocessing successfully applied!")
                st.dataframe(preprocessed_df.head(5), use_container_width=True)
                st.info("You can now proceed to the **Feature Engineering** page in the sidebar.")
            except Exception as e:
                st.error(f"Error applying preprocessing: {str(e)}")
                logger.error("Failed to run preprocessing: %s", str(e), exc_info=True)
                
    elif get_state("preprocessed_data") is not None:
        st.success("Previously preprocessed data exists in session state.")
        st.dataframe(get_state("preprocessed_data").head(5), use_container_width=True)



render_preprocessing_page()
