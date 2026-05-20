"""SmartML Studio — Dataset Analysis page."""

import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config.logging_config import setup_logging
from src.profiling.dataset_profiler import DatasetProfiler
from utils.session_state import get_state, set_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_dataset_analysis_page() -> None:
    """Render the Dataset Analysis page."""
    st.header("Dataset Analysis")

    df = get_state("dataset")
    target_col = get_state("target_column")
    problem_type = get_state("problem_type")

    # Prerequisite guard
    if df is None:
        st.warning("Please upload a dataset first in the Dataset Upload page.")
        st.stop()

    # Calculate profile if not already cached in session state
    profile = get_state("profile")
    if profile is None:
        with st.spinner("Analyzing dataset statistical properties..."):
            profiler = DatasetProfiler(df, target_col)
            profile = profiler.compute_profile()
            set_state("profile", profile)

    # ── Display Profile Dashboard ──────────────────────────────────────
    st.markdown(
        f"Analyzing dataset **{get_state('dataset_name')}** for **{problem_type}** "
        f"predicting target **{target_col}**."
    )

    tab_overview, tab_missing, tab_dtypes, tab_outliers, tab_corr, tab_target = st.tabs(
        [
            "Overview",
            "Missing Values",
            "Columns & Cardinality",
            "Outliers & Skewness",
            "Correlations",
            "Target Variable",
        ]
    )

    # ── Overview Tab ──────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Dataset Summary")
        rows, cols = profile["shape"]
        duplicates = profile["duplicates"]
        mem_mb = profile["memory_bytes"] / (1024 * 1024)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{rows:,}")
        col2.metric("Columns", f"{cols:,}")
        col3.metric("Duplicate Rows", f"{duplicates:,}")
        col4.metric("Memory Footprint", f"{mem_mb:.2f} MB")

        # Warnings panel
        warnings = []
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate rows. Consider dropping them in Preprocessing.")
        
        # Missing values warnings
        total_missing = sum(profile["missing_values"].values())
        if total_missing > 0:
            warnings.append(f"Dataset contains {total_missing} missing values across multiple columns.")

        # Outliers warnings
        total_outliers = sum(info["count"] for info in profile["outliers"].values())
        if total_outliers > 0:
            warnings.append(f"Found {total_outliers} outliers across numeric features using the IQR threshold.")

        if warnings:
            st.info("**Key Observations:**\n" + "\n".join([f"- {w}" for w in warnings]))
        else:
            st.success("Dataset looks extremely clean! No duplicates, missing values, or outliers detected.")

    # ── Missing Values Tab ────────────────────────────────────────────
    with tab_missing:
        st.subheader("Missingness Profile")
        missing_df = pd.DataFrame({
            "Missing Count": profile["missing_values"].values(),
            "Percentage (%)": profile["missing_pct"].values()
        }, index=profile["missing_values"].keys())
        
        # Filter to columns with missing values or show all if clean
        columns_with_missing = missing_df[missing_df["Missing Count"] > 0]
        
        if len(columns_with_missing) > 0:
            st.dataframe(columns_with_missing.sort_values(by="Missing Count", ascending=False), use_container_width=True)
            
            # Missingness Chart
            fig = px.bar(
                columns_with_missing.reset_index(),
                x="index",
                y="Percentage (%)",
                title="Missing Value Percentage per Column",
                labels={"index": "Column", "Percentage (%)": "Missing %"},
                color_discrete_sequence=["#FF4B4B"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Excellent! No missing values detected in the entire dataset.")

    # ── Columns & Cardinality Tab ─────────────────────────────────────
    with tab_dtypes:
        st.subheader("Column Data Types & Cardinality")
        
        cardinality_df = pd.DataFrame({
            "DataType": profile["dtypes"].values(),
            "Unique Values (Cardinality)": profile["cardinality"].values(),
            "Kind": ["Numeric" if c in profile["numeric_columns"] else "Categorical" for c in profile["dtypes"].keys()]
        }, index=profile["dtypes"].keys())
        
        st.dataframe(cardinality_df, use_container_width=True)

        # High cardinality categorical warning
        high_card_cols = [
            col for col in profile["categorical_columns"] 
            if profile["cardinality"][col] > 20 and col != target_col
        ]
        if high_card_cols:
            st.warning(
                f"The following categorical columns have high cardinality (>20 unique values) "
                f"which might require target encoding or grouping in Preprocessing: **{', '.join(high_card_cols)}**"
            )

    # ── Outliers & Skewness Tab ───────────────────────────────────────
    with tab_outliers:
        st.subheader("Skewness & Outliers (Numeric Features)")
        
        numeric_cols = profile["numeric_columns"]
        if not numeric_cols:
            st.info("No numeric features to analyse.")
        else:
            outlier_counts = [profile["outliers"][col]["count"] for col in numeric_cols]
            outlier_pct = [(profile["outliers"][col]["count"] / rows) * 100 for col in numeric_cols]
            skews = [profile["skewness"][col] for col in numeric_cols]

            outliers_df = pd.DataFrame({
                "Skewness": skews,
                "Outlier Count": outlier_counts,
                "Outlier Percentage (%)": outlier_pct
            }, index=numeric_cols)

            st.dataframe(outliers_df.round(3), use_container_width=True)

            # High skew warnings
            high_skew_cols = [col for col in numeric_cols if abs(profile["skewness"][col]) > 1.0]
            if high_skew_cols:
                st.warning(
                    f"Highly skewed features detected (absolute skew > 1.0): **{', '.join(high_skew_cols)}**. "
                    "Log transformations or Box-Cox transformations are recommended."
                )

    # ── Correlations Tab ──────────────────────────────────────────────
    with tab_corr:
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = profile["correlation_matrix"]
        
        if not corr_matrix:
            st.info("Not enough numeric features to calculate a correlation matrix.")
        else:
            corr_df = pd.DataFrame(corr_matrix)
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                title="Spearman/Pearson Correlation Coefficients",
                height=500,
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Target Variable Tab ───────────────────────────────────────────
    with tab_target:
        st.subheader(f"Target Column: {target_col}")

        if problem_type == "Classification":
            balance = profile["class_balance"]
            balance_df = pd.DataFrame({
                "Count": balance.values(),
                "Percentage (%)": [(v / rows) * 100 for v in balance.values()]
            }, index=balance.keys())
            st.dataframe(balance_df.round(2), use_container_width=True)

            # Bar Chart for class balance
            fig = px.bar(
                balance_df.reset_index(),
                x="index",
                y="Count",
                title="Class Balance Distribution",
                labels={"index": "Class/Label", "Count": "Frequency"},
                color="index"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Class imbalance alert
            if len(balance) > 1:
                vals = list(balance.values())
                ratio = max(vals) / min(vals) if min(vals) > 0 else float("inf")
                if ratio > 3.0:
                    st.warning(
                        f"Significant class imbalance detected (ratio {ratio:.2f} > 3.0). "
                        "We recommend stratifying splits or adjusting class weights."
                    )
        
        elif problem_type == "Regression":
            stats = profile["target_summary"]
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{stats['mean']:.4f}")
                col2.metric("Median", f"{stats['median']:.4f}")
                col3.metric("Min", f"{stats['min']:.4f}")
                col4.metric("Max", f"{stats['max']:.4f}")

                # Histogram for target
                fig = px.histogram(
                    df,
                    x=target_col,
                    title=f"Distribution of Target Column: {target_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Time Series
            st.write("Time Series Target detected. Summary Statistics:")
            st.write(df[target_col].describe())
            fig = px.line(
                df,
                y=target_col,
                title=f"Time Series Plot of target: {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)



render_dataset_analysis_page()
