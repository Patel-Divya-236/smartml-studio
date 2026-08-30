"""Smart Visualization Advisor module.

Generates ranked, confidence-scored visualization recommendations
based on dataset profile characteristics.
"""

import logging
from typing import Any

from src.advisors.base import BaseAdvisor, Recommendation
from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class VisualizationAdvisor(BaseAdvisor):
    """Recommends visualizations based on dataset profile."""

    def recommend(self, profile: dict[str, Any]) -> list[Recommendation]:
        """Generate visualization recommendations based on dataset profile.

        Args:
            profile: Dictionary produced by DatasetProfiler.compute_profile().

        Returns:
            List of Recommendation objects sorted by confidence_score descending.
        """
        logger.info("Generating visualization recommendations...")
        recs = []

        # 1. Heatmap for numeric correlations
        numeric_cols = profile.get("numeric_columns", [])
        if len(numeric_cols) > 1:
            recs.append(
                Recommendation(
                    label="Feature Correlation Heatmap",
                    confidence_score=0.95,
                    reason=f"Found {len(numeric_cols)} numeric columns, making correlation analysis high value.",
                    why_explanation="A correlation heatmap displays pairwise linear correlation coefficients. "
                                    "It helps you identify multicollinearity (highly correlated features) and "
                                    "see which features might have strong relationships with the target variable.",
                    category="Heatmap",
                    metadata={"chart_type": "correlation_heatmap"}
                )
            )

        # 2. Target Distribution
        # The task type comes from the user's confirmed choice, carried through the
        # profile. A continuous target still has a populated target_summary, so the
        # histogram branch below fires for regression.
        class_balance = profile.get("class_balance", {})
        target_summary = profile.get("target_summary", {})

        has_class_target = profile.get("problem_type") == "Classification" and len(class_balance) > 0
        has_numeric_target = len(target_summary) > 0

        if has_class_target:
            num_classes = len(class_balance)
            if num_classes <= SETTINGS.MAX_CATEGORIES_FOR_PIE:
                recs.append(
                    Recommendation(
                        label="Target Class Balance Pie Chart",
                        confidence_score=0.90,
                        reason=f"Target column has {num_classes} unique categories (<= {SETTINGS.MAX_CATEGORIES_FOR_PIE}).",
                        why_explanation="A pie chart provides an intuitive visual breakdown of category proportions. "
                                        "It highlights immediately if there is a severe class imbalance problem "
                                        "that requires stratified splitting or rescaled class weights during training.",
                        category="Pie",
                        metadata={"chart_type": "target_pie_chart"}
                    )
                )
            else:
                recs.append(
                    Recommendation(
                        label="Target Class Balance Bar Chart",
                        confidence_score=0.88,
                        reason=f"Target column is categorical with {num_classes} categories (> {SETTINGS.MAX_CATEGORIES_FOR_PIE}).",
                        why_explanation="A horizontal or vertical bar chart is superior to a pie chart when there are "
                                        "many categories. It avoids visual clutter while clearly showing class frequency.",
                        category="Bar",
                        metadata={"chart_type": "target_bar_chart"}
                    )
                )
        elif has_numeric_target:
            recs.append(
                Recommendation(
                    label="Target Distribution Histogram",
                    confidence_score=0.92,
                    reason="Target is numeric, which is ideal for a distribution plot.",
                    why_explanation="A distribution histogram with an overlaid box plot helps check normality, "
                                    "detect outliers, and identify skewness in the target variable, which is crucial "
                                    "for model assumptions (e.g. linear regression assumes normal residuals).",
                    category="Histogram",
                    metadata={"chart_type": "target_histogram"}
                )
            )

        # 3. Time Series line plot
        # Let's check if the problem type or columns indicate a datetime
        # We can add a Time Series Line Plot recommendation if TS or date columns exist
        categorical_cols = profile.get("categorical_columns", [])
        date_cols = [col for col, dtype in profile.get("dtypes", {}).items() if "date" in dtype or "time" in dtype]
        if date_cols:
            recs.append(
                Recommendation(
                    label="Time Series Target Plot",
                    confidence_score=0.95,
                    reason=f"Detected date/time columns: {', '.join(date_cols)}.",
                    why_explanation="Plotting target values over time helps detect trends, seasonal patterns, "
                                    "and structural breaks in the data.",
                    category="Line",
                    metadata={"chart_type": "time_series_line", "time_col": date_cols[0]}
                )
            )

        # 4. Outlier scatter plots or strong correlations
        corr_matrix = profile.get("correlation_matrix", {})
        strong_pairs = []
        seen = set()
        for col1, targets in corr_matrix.items():
            for col2, val in targets.items():
                if col1 != col2 and abs(val) >= SETTINGS.CORRELATION_STRONG_THRESHOLD:
                    pair = tuple(sorted([col1, col2]))
                    if pair not in seen:
                        seen.add(pair)
                        strong_pairs.append((col1, col2, val))

        for col1, col2, val in strong_pairs[:3]:  # Suggest up to top 3
            recs.append(
                Recommendation(
                    label=f"Bivariate Scatter Plot: {col1} vs {col2}",
                    confidence_score=0.85,
                    reason=f"Strong linear correlation detected between {col1} and {col2} (r = {val:.2f}).",
                    why_explanation="A scatter plot displays the relationship between two continuous variables. "
                                    "It helps confirm if the relationship is truly linear, quadratic, or non-linear, "
                                    "and shows if any specific data points act as leverage points/outliers.",
                    category="Scatter",
                    metadata={"chart_type": "bivariate_scatter", "col_x": col1, "col_y": col2}
                )
            )

        # 5. Skewed column distributions (Histograms)
        skewness_dict = profile.get("skewness", {})
        skewed_cols = [col for col, skew in skewness_dict.items() if abs(skew) > SETTINGS.SKEWNESS_THRESHOLD]
        for col in skewed_cols[:2]:  # Limit to top 2 skewed
            recs.append(
                Recommendation(
                    label=f"Skewed Feature Histogram: {col}",
                    confidence_score=0.80,
                    reason=f"Feature '{col}' is highly skewed (skew = {skewness_dict[col]:.2f}).",
                    why_explanation=f"A histogram for '{col}' reveals the shape of its distribution. "
                                    "This visual justification guides decisions on applying transformations "
                                    "(e.g., Log or Box-Cox) to normalize the feature's scale for modeling.",
                    category="Histogram",
                    metadata={"chart_type": "feature_histogram", "column": col}
                )
            )

        # 6. Low cardinality features (Bar Chart)
        for col in categorical_cols:
            card = profile.get("cardinality", {}).get(col, 0)
            if 1 < card <= SETTINGS.MAX_CATEGORIES_FOR_BAR:
                recs.append(
                    Recommendation(
                        label=f"Categorical Frequency: {col}",
                        confidence_score=0.75,
                        reason=f"Categorical column '{col}' has low cardinality ({card} unique values).",
                        why_explanation=f"A frequency bar chart for '{col}' helps visualize class proportions "
                                        "and find rare categories that might cause issues during cross-validation splits.",
                        category="Bar",
                        metadata={"chart_type": "categorical_bar", "column": col}
                    )
                )

        # Sort recommendations by confidence descending
        recs.sort(key=lambda r: r.confidence_score, reverse=True)
        return recs
