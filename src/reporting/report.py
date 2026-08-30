"""Markdown evaluation report builder.

Moved out of `pages/11_download.py` during the React/FastAPI migration: the report is
domain output, not view code, and the API serves it as a file download.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_evaluation_report(
    dataset_name: str,
    target_col: str,
    problem_type: str,
    profile: dict,
    model_comparison: pd.DataFrame,
    report_state: dict,
    executive_summary: str | None = None,
) -> str:
    """Compile the entire pipeline run info into a Markdown Report.

    Args:
        executive_summary: Optional LLM-written opening summary. When None the
            section is omitted entirely, so the report still builds without an LLM.
    """
    shape = profile.get("shape", (0, 0))
    rows_count = shape[0]
    cols_count = shape[1]
    missing_dict = profile.get("missing_values", {})
    missing_count = sum(missing_dict.values()) if isinstance(missing_dict, dict) else 0
    duplicate_count = profile.get("duplicates", 0)


    summary_section = ""
    if executive_summary:
        summary_section = f"""
## Executive Summary

{executive_summary}
"""

    report_md = f"""# SmartML Studio — ML Pipeline Evaluation Report
{summary_section}
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
