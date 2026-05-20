"""SmartML Studio — Smart Visualization page."""

import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config.logging_config import setup_logging
from src.advisors.visualization_advisor import VisualizationAdvisor
from utils.session_state import get_state, set_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def render_visualization_page() -> None:
    """Render the Smart Visualization page."""
    st.header("Smart Visualization")

    df = get_state("dataset")
    profile = get_state("profile")
    target_col = get_state("target_column")

    # Prerequisite guard
    if df is None or profile is None:
        st.warning("Please complete Dataset Analysis first.")
        st.stop()

    # Retrieve or generate recommendations
    recs = get_state("viz_recommendations")
    if recs is None:
        advisor = VisualizationAdvisor()
        recs = advisor.recommend(profile)
        set_state("viz_recommendations", recs)

    st.markdown(
        """
        The Visualization Advisor has analyzed your dataset properties and recommended
        the most useful charts. Review the suggestions, check/uncheck as desired,
        optionally create custom charts, and click **Generate Dashboard** at the bottom.
        """
    )

    # ── Checklist UI ──────────────────────────────────────────────────
    st.subheader("Recommended Visualizations")
    selected_recs = []

    for i, rec in enumerate(recs):
        col_chk, col_why = st.columns([0.7, 0.3])
        with col_chk:
            # Rating label
            label_text = f"**{rec.label}** (Rating: {rec.star_rating}/5 | {rec.confidence_score * 100:.0f}% confidence) — *{rec.reason}*"
            is_selected = st.checkbox(label_text, value=True, key=f"chk_rec_{i}")
            if is_selected:
                selected_recs.append(rec)
        with col_why:
            with st.expander("Why?"):
                st.write(rec.why_explanation)

    st.divider()

    # ── Custom Visualizations ──────────────────────────────────────────
    st.subheader("Create Custom Chart")
    with st.expander("Configure a manual plot"):
        col_type, col_x_sel, col_y_sel = st.columns(3)
        with col_type:
            custom_type = st.selectbox(
                "Chart Type:",
                options=["None", "Scatter Plot", "Histogram", "Line Plot", "Bar Chart", "Box Plot"]
            )
        with col_x_sel:
            custom_x = st.selectbox("X Axis Column:", options=list(df.columns), key="custom_x_widget")
        with col_y_sel:
            custom_y = st.selectbox("Y Axis Column:", options=list(df.columns), key="custom_y_widget")

        add_btn = st.button("Add Chart to Queue")
        if add_btn and custom_type != "None":
            # Build custom recommendation
            custom_rec = None
            if custom_type == "Scatter Plot":
                custom_rec = Recommendation(
                    label=f"Custom Scatter: {custom_x} vs {custom_y}",
                    confidence_score=1.0,
                    star_rating=5,
                    reason="User-configured custom plot",
                    why_explanation="User custom visualization.",
                    category="Scatter",
                    metadata={"chart_type": "bivariate_scatter", "col_x": custom_x, "col_y": custom_y}
                )
            elif custom_type == "Histogram":
                custom_rec = Recommendation(
                    label=f"Custom Histogram: {custom_x}",
                    confidence_score=1.0,
                    star_rating=5,
                    reason="User-configured custom plot",
                    why_explanation="User custom visualization.",
                    category="Histogram",
                    metadata={"chart_type": "feature_histogram", "column": custom_x}
                )
            elif custom_type == "Line Plot":
                custom_rec = Recommendation(
                    label=f"Custom Line: {custom_x} vs {custom_y}",
                    confidence_score=1.0,
                    star_rating=5,
                    reason="User-configured custom plot",
                    why_explanation="User custom visualization.",
                    category="Line",
                    metadata={"chart_type": "time_series_line", "time_col": custom_x, "target_col_override": custom_y}
                )
            elif custom_type == "Bar Chart":
                custom_rec = Recommendation(
                    label=f"Custom Bar: {custom_x}",
                    confidence_score=1.0,
                    star_rating=5,
                    reason="User-configured custom plot",
                    why_explanation="User custom visualization.",
                    category="Bar",
                    metadata={"chart_type": "categorical_bar", "column": custom_x}
                )
            elif custom_type == "Box Plot":
                custom_rec = Recommendation(
                    label=f"Custom Box Plot: {custom_x}",
                    confidence_score=1.0,
                    star_rating=5,
                    reason="User-configured custom plot",
                    why_explanation="User custom visualization.",
                    category="Box",
                    metadata={"chart_type": "box_plot", "column": custom_x}
                )

            if custom_rec:
                recs.append(custom_rec)
                set_state("viz_recommendations", recs)
                st.success(f"Added custom chart: '{custom_rec.label}'! Please refresh checklist state.")
                st.rerun()

    st.divider()

    # ── Gated Chart Generation ──────────────────────────────────────────
    generate_btn = st.button("Generate Dashboard", type="primary")

    # Store generating state
    if "dashboard_generated" not in st.session_state:
        st.session_state.dashboard_generated = False

    if generate_btn:
        st.session_state.dashboard_generated = True

    if st.session_state.dashboard_generated and selected_recs:
        st.subheader("Visualization Dashboard")
        
        for idx, rec in enumerate(selected_recs):
            st.markdown(f"#### {rec.label}")
            meta = rec.metadata
            chart_type = meta.get("chart_type")

            try:
                if chart_type == "correlation_heatmap":
                    corr_matrix = profile["correlation_matrix"]
                    corr_df = pd.DataFrame(corr_matrix)
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.index,
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1
                    ))
                    fig.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_heatmap_{idx}")

                elif chart_type == "target_pie_chart":
                    class_balance = profile["class_balance"]
                    fig = px.pie(
                        names=list(class_balance.keys()),
                        values=list(class_balance.values()),
                        title="Target Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_pie_{idx}")

                elif chart_type == "target_bar_chart":
                    class_balance = profile["class_balance"]
                    fig = px.bar(
                        x=list(class_balance.keys()),
                        y=list(class_balance.values()),
                        labels={"x": "Class", "y": "Frequency"}
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_bar_t_{idx}")

                elif chart_type == "target_histogram":
                    fig = px.histogram(df, x=target_col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_hist_t_{idx}")

                elif chart_type == "time_series_line":
                    time_col = meta.get("time_col")
                    y_col = meta.get("target_col_override", target_col)
                    fig = px.line(df, x=time_col, y=y_col)
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_line_{idx}")

                elif chart_type == "bivariate_scatter":
                    col_x = meta.get("col_x")
                    col_y = meta.get("col_y")
                    fig = px.scatter(df, x=col_x, y=col_y, trendline="ols")
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_scatter_{idx}")

                elif chart_type == "feature_histogram":
                    column = meta.get("column")
                    fig = px.histogram(df, x=column, marginal="rug")
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_hist_{idx}")

                elif chart_type == "categorical_bar":
                    column = meta.get("column")
                    counts = df[column].value_counts().reset_index()
                    fig = px.bar(counts, x=column, y="count")
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_bar_{idx}")
                    
                elif chart_type == "box_plot":
                    column = meta.get("column")
                    fig = px.box(df, y=column)
                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_box_{idx}")

            except Exception as e:
                st.error(f"Failed to render chart '{rec.label}': {str(e)}")
                logger.error("Failed to render chart %s: %s", rec.label, str(e), exc_info=True)
            
            st.divider()
    elif st.session_state.dashboard_generated:
        st.info("No charts selected. Please check at least one visualization recommendation.")



render_visualization_page()
