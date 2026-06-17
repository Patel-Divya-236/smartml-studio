import logging
import pickle
import pandas as pd
import streamlit as st

from config.logging_config import setup_logging
from utils.session_state import get_state
from utils.styling import apply_custom_theme

setup_logging()
logger = logging.getLogger(__name__)
apply_custom_theme()



def build_evaluation_report(
    dataset_name: str,
    target_col: str,
    problem_type: str,
    profile: dict,
    model_comparison: pd.DataFrame,
    report_state: dict
) -> str:
    """Compile the entire pipeline run info into a Markdown Report."""
    shape = profile.get("shape", (0, 0))
    rows_count = shape[0]
    cols_count = shape[1]
    missing_dict = profile.get("missing_values", {})
    missing_count = sum(missing_dict.values()) if isinstance(missing_dict, dict) else 0
    duplicate_count = profile.get("duplicates", 0)


    report_md = f"""# SmartML Studio — ML Pipeline Evaluation Report

## 1. Project Context
- **Original Dataset Filename:** {dataset_name}
- **Target Feature (Target Column):** `{target_col}`
- **Task Type:** {problem_type}

## 2. Dataset Profile Summary
- **Total Sample Count:** {rows_count} rows
- **Feature Cardinality:** {cols_count} columns
- **Missing Value Count:** {missing_count} cells
- **Duplicate Rows:** {duplicate_count}

## 3. Preprocessing Configs
- Imputation, scaling, and encoding recommendations were computed per feature and accepted/adapted.
"""

    if "prediction_mode" in report_state:
        report_md += f"""
## 4. Final Prediction & Strategy Selection
- **Selected Strategy:** {report_state.get('selected_model', 'N/A')}
- **Ensemble Setup:**
  - Voting method: {report_state.get('ensemble_voting', 'N/A')}
  - Combined Models: {", ".join(report_state.get('ensemble_models', [])) or 'None'}
"""

    report_md += "\n## 5. Model Comparisons & Performance Summary\n"
    if model_comparison is not None:
        # Convert df to markdown table without depending on external tabulate package
        df_rounded = model_comparison.round(4)
        headers = list(df_rounded.columns)
        report_md += "| " + " | ".join(headers) + " |\n"
        report_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df_rounded.iterrows():
            row_vals = [str(val) for val in row.values]
            report_md += "| " + " | ".join(row_vals) + " |\n"
    else:
        report_md += "*No models evaluated.*"


    report_md += """

---
*Report automatically compiled by SmartML Studio.*
"""
    return report_md


def render_download_and_report_page() -> None:
    """Render the Download & Report page."""
    st.header("Download & Report")

    # Prerequisite check
    trained_models = get_state("trained_models")
    if trained_models is None:
        st.warning("Please train models first in the Model Training page.")
        st.stop()

    dataset_name = get_state("dataset_name") or "dataset"
    target_column = get_state("target_column")
    problem_type = get_state("problem_type")
    profile = get_state("profile") or {}
    model_comparison = get_state("model_comparison")
    predictions = get_state("predictions")
    ensemble = get_state("ensemble")
    report_state = get_state("evaluation_report") or {}

    st.markdown(
        """
        Finalise your machine learning pipeline. Export predictions, download the trained models
        as serialized binaries, or generate a comprehensive PDF/Markdown summary of your pipeline runs.
        """
    )

    # ── Part 1: Download Predictions ───────────────────────────────────
    st.subheader("1. Export Prediction Results")
    if predictions is not None:
        csv_data = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_data,
            file_name=f"{dataset_name}_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("To download test set predictions, please generate predictions in the **Prediction** page first.")

    st.divider()

    # ── Part 2: Download Model Binaries ────────────────────────────────
    st.subheader("2. Export Trained Model Artifacts")
    st.write("Serialize the estimators into standard Python pickle binaries.")
    
    # Model option selection
    model_options = list(trained_models.keys())
    if ensemble is not None:
        model_options.append("Custom Hybrid Ensemble")

    selected_model_to_save = st.selectbox(
        "Select model to export:",
        options=model_options
    )

    if selected_model_to_save:
        try:
            if selected_model_to_save == "Custom Hybrid Ensemble":
                model_obj = ensemble
                filename = "hybrid_ensemble.pkl"
            else:
                model_obj = trained_models[selected_model_to_save]["instance"]
                filename = f"{selected_model_to_save.replace(' ', '_').lower()}.pkl"

            model_bytes = pickle.dumps(model_obj)
            st.download_button(
                label=f"Download {selected_model_to_save} (.pkl)",
                data=model_bytes,
                file_name=filename,
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Could not serialize model: {str(e)}")
            logger.error("Serialization failed for %s: %s", selected_model_to_save, str(e), exc_info=True)

    st.divider()

    # ── Part 3: Compile Report ────────────────────────────────────────
    st.subheader("3. Pipeline Summary Report")
    
    if st.button("Generate Report Preview", type="secondary"):
        report_content = build_evaluation_report(
            dataset_name,
            target_column,
            problem_type,
            profile,
            model_comparison,
            report_state
        )
        
        st.markdown("### Report Preview")
        st.markdown(report_content)
        
        st.download_button(
            label="Download Evaluation Report (.md)",
            data=report_content.encode("utf-8"),
            file_name="smartml_evaluation_report.md",
            mime="text/markdown"
        )



render_download_and_report_page()

