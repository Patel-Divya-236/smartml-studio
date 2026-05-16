"""SmartML Studio — Dataset Upload page."""

import logging
import pandas as pd
import streamlit as st

from config.logging_config import setup_logging
from config.settings import SETTINGS
from utils.session_state import get_state, set_state, reset_downstream
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



@st.cache_data(show_spinner="Loading data...")
def load_data(file) -> pd.DataFrame:
    """Load dataset from uploaded file.

    Supports CSV and Excel formats.
    """
    name = file.name
    logger.info("Loading file: %s", name)
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        logger.info("Successfully loaded file: %s with shape %s", name, df.shape)
        return df
    except Exception as e:
        logger.error("Failed to load file %s: %s", name, str(e), exc_info=True)
        raise e


def detect_problem_type(df: pd.DataFrame, target_col: str) -> tuple[str, float, str]:
    """Auto-detect the problem type (Classification/Regression/Time Series) for the target.

    Returns:
        tuple containing (problem_type, confidence_score, explanation_reason)
    """
    col_data = df[target_col]
    dtype_str = str(col_data.dtype)

    # 1. Datetime check -> Time Series
    if "datetime" in dtype_str or "date" in dtype_str or col_data.apply(lambda x: isinstance(x, pd.Timestamp)).all():
        return (
            "Time Series",
            0.90,
            f"Target column '{target_col}' has a datetime dtype ({dtype_str}), suggesting a forecasting task."
        )

    # If it is numeric
    if pd.api.types.is_numeric_dtype(col_data):
        unique_count = col_data.nunique()
        if unique_count <= SETTINGS.MAX_UNIQUE_VALUES_FOR_CLASSIFICATION:
            return (
                "Classification",
                0.85,
                f"Target column '{target_col}' is numeric but has only {unique_count} unique values (<= {SETTINGS.MAX_UNIQUE_VALUES_FOR_CLASSIFICATION}), indicating discrete classes."
            )
        else:
            return (
                "Regression",
                0.90,
                f"Target column '{target_col}' is numeric and has a high number of unique values ({unique_count}), indicating a continuous target."
            )

    # Object / String / Categorical / Bool
    unique_count = col_data.nunique()
    return (
        "Classification",
        0.95,
        f"Target column '{target_col}' is categorical/object type with {unique_count} unique classes, suggesting a classification task."
    )


def render_dataset_upload_page() -> None:
    """Render the Dataset Upload page."""
    st.header("Dataset Upload")

    st.markdown(
        """
        Upload your tabular dataset (CSV or Excel) below. Once uploaded, select your
        **Target Column** (the column you want to predict). The system will automatically
        detect the machine learning problem type, which you can review and override.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload a structured dataset with column headers."
    )

    if uploaded_file is not None:
        # If a new file is uploaded, reset downstream state
        current_dataset_name = get_state("dataset_name")
        if current_dataset_name != uploaded_file.name:
            logger.info("New file detected: %s (previously: %s). Resetting session state.", uploaded_file.name, current_dataset_name)
            set_state("dataset", None)
            set_state("dataset_name", uploaded_file.name)
            reset_downstream("dataset_name")

        try:
            df = load_data(uploaded_file)
            set_state("dataset", df)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

        # Display details
        rows, cols = df.shape
        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{rows:,}")
        col2.metric("Columns", f"{cols:,}")
        col3.metric("Memory Usage", f"{mem_usage:.2f} MB")

        st.subheader("Data Preview (First 5 rows)")
        st.dataframe(df.head(5), use_container_width=True)

        st.divider()

        # Target Column selector
        st.subheader("Target Column & Problem Type Selection")
        
        column_list = list(df.columns)
        
        # Preserve previous target selection if still valid
        prev_target = get_state("target_column")
        default_index = column_list.index(prev_target) if prev_target in column_list else 0

        target_col = st.selectbox(
            "Select the Target Column (what you want to predict):",
            options=column_list,
            index=default_index,
            key="target_col_select_widget"
        )

        # Trigger reset downstream if target changes
        if prev_target != target_col:
            logger.info("Target column changed from %s to %s. Resetting downstream state.", prev_target, target_col)
            set_state("target_column", target_col)
            reset_downstream("target_column")

        # Auto-detect problem type
        detected_type, conf, reason = detect_problem_type(df, target_col)

        # Rating for confidence
        rating = max(1, min(5, round(conf * 5)))
        st.markdown(
            f"**Auto-Detected Problem Type:** `{detected_type}` (Rating: {rating}/5 | {conf * 100:.0f}% confidence)"
        )
        with st.expander("Why?"):
            st.write(reason)

        # User Decides (Accept or Override)
        problem_types = ["Classification", "Regression", "Time Series"]
        prev_prob_type = get_state("problem_type")
        
        # Default to detected type unless user overrides
        initial_prob_index = problem_types.index(detected_type)
        if prev_prob_type in problem_types:
            initial_prob_index = problem_types.index(prev_prob_type)

        selected_prob_type = st.selectbox(
            "Confirm or override the problem type:",
            options=problem_types,
            index=initial_prob_index,
            key="prob_type_select_widget"
        )

        if prev_prob_type != selected_prob_type:
            logger.info("Problem type set to: %s", selected_prob_type)
            set_state("problem_type", selected_prob_type)
            reset_downstream("problem_type")

        st.success(
            f"Configured! Target: **{target_col}** | Problem Type: **{selected_prob_type}**. "
            "You can now proceed to the **Dataset Analysis** page in the sidebar."
        )

    else:
        st.info("Please upload a CSV or Excel dataset to begin.")
        # Clear state if file is removed
        if get_state("dataset") is not None:
            logger.info("File removed by user. Resetting session state.")
            set_state("dataset", None)
            set_state("dataset_name", None)
            reset_downstream("dataset_name")



render_dataset_upload_page()
