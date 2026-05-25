"""Smart Preprocessing Advisor module.

Recommends per-column imputation, encoding, scaling, and outlier
handling strategies based on dataset profile.
"""

import logging
from typing import Any

from src.advisors.base import BaseAdvisor, Recommendation
from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class PreprocessingAdvisor(BaseAdvisor):
    """Recommends preprocessing strategies per column."""

    def recommend(self, profile: dict[str, Any]) -> list[Recommendation]:
        """Generate preprocessing recommendations.

        Args:
            profile: Dictionary produced by DatasetProfiler.compute_profile().

        Returns:
            List of Recommendation objects sorted by confidence_score descending.
        """
        logger.info("Generating preprocessing recommendations...")
        recs = []
        
        missing_pcts = profile.get("missing_pct", {})
        dtypes = profile.get("dtypes", {})
        cardinalities = profile.get("cardinality", {})
        skewness_dict = profile.get("skewness", {})
        outliers_dict = profile.get("outliers", {})
        numeric_cols = profile.get("numeric_columns", [])
        categorical_cols = profile.get("categorical_columns", [])
        
        # ── 1. Imputation Recommendations ────────────────────────────────
        for col, pct in missing_pcts.items():
            if pct > 0:
                is_num = col in numeric_cols
                if pct > SETTINGS.MISSING_HIGH_PCT:
                    recs.append(
                        Recommendation(
                            label=f"Drop Column: {col}",
                            confidence_score=0.85,
                            reason=f"Column '{col}' has {pct:.1f}% missing values (> {SETTINGS.MISSING_HIGH_PCT}%).",
                            why_explanation=f"When a column has a very high proportion of missing values, "
                                            "imputation can introduce excessive noise or bias. It is usually "
                                            "better to drop the feature entirely.",
                            category="imputation",
                            metadata={"column": col, "action": "drop"}
                        )
                    )
                elif is_num:
                    if pct <= SETTINGS.MISSING_LOW_PCT:
                        recs.append(
                            Recommendation(
                                label=f"Median Imputer for: {col}",
                                confidence_score=0.90,
                                reason=f"Column '{col}' has low missingness ({pct:.1f}% <= {SETTINGS.MISSING_LOW_PCT}%).",
                                why_explanation="For low missingness, a simple median imputation is fast, robust, "
                                                "and preserves numerical stability without overfitting.",
                                category="imputation",
                                metadata={"column": col, "action": "median"}
                            )
                        )
                    else:
                        recs.append(
                            Recommendation(
                                label=f"KNN Imputer for: {col}",
                                confidence_score=0.85,
                                reason=f"Column '{col}' has medium missingness ({pct:.1f}%).",
                                why_explanation="KNN Imputation fills missing values using distance-weighted averages "
                                                "of neighboring samples, preserving multi-variable relationships better "
                                                "than simple univariate median.",
                                category="imputation",
                                metadata={"column": col, "action": "knn"}
                            )
                        )
                else:
                    # Categorical missing values
                    recs.append(
                        Recommendation(
                            label=f"Mode (Most Frequent) Imputer for: {col}",
                            confidence_score=0.90,
                            reason=f"Categorical column '{col}' has {pct:.1f}% missing values.",
                            why_explanation="Mode imputation replaces missing values in a categorical feature "
                                            "with the most frequent class, maintaining the string format safely.",
                            category="imputation",
                            metadata={"column": col, "action": "mode"}
                        )
                    )

        # ── 2. Encoding Recommendations ──────────────────────────────────
        for col in categorical_cols:
            card = cardinalities.get(col, 0)
            if card <= 10:
                recs.append(
                    Recommendation(
                        label=f"One-Hot Encode: {col}",
                        confidence_score=0.95,
                        reason=f"Categorical column '{col}' has low cardinality ({card} unique values).",
                        why_explanation="One-Hot Encoding converts categorical levels into binary columns. "
                                        "Since cardinality is low, it won't explode the feature space, allowing linear "
                                        "and tree-based models to split on specific categories easily.",
                        category="encoding",
                        metadata={"column": col, "action": "onehot"}
                    )
                )
            else:
                recs.append(
                    Recommendation(
                        label=f"Ordinal / Label Encode: {col}",
                        confidence_score=0.85,
                        reason=f"Categorical column '{col}' has high cardinality ({card} unique values).",
                        why_explanation="One-hot encoding high-cardinality features causes the curse of dimensionality. "
                                        "Ordinal or Label Encoding maps categories to integer values, keeping the feature "
                                        "representation compact.",
                        category="encoding",
                        metadata={"column": col, "action": "ordinal"}
                    )
                )

        # ── 3. Scaling & Transformation Recommendations ──────────────────
        for col in numeric_cols:
            skew = skewness_dict.get(col, 0.0)
            outliers = outliers_dict.get(col, {}).get("count", 0)
            
            if abs(skew) > SETTINGS.SKEWNESS_THRESHOLD:
                recs.append(
                    Recommendation(
                        label=f"Log1p Transform: {col}",
                        confidence_score=0.85,
                        reason=f"Column '{col}' is highly skewed (skew = {skew:.2f}).",
                        why_explanation="Log1p transform (log(1 + x)) compresses the tail of skewed distributions, "
                                        "helping algorithms that assume normally distributed errors (e.g. Linear Models, SVM) "
                                        "to converge and perform better.",
                        category="scaling",
                        metadata={"column": col, "action": "log1p"}
                    )
                )
            elif outliers > 0:
                recs.append(
                    Recommendation(
                        label=f"Robust Scale: {col}",
                        confidence_score=0.90,
                        reason=f"Column '{col}' has {outliers} outliers using IQR.",
                        why_explanation="StandardScaler is highly sensitive to outliers because it uses the mean and variance. "
                                        "RobustScaler uses interquartile range (IQR) and median, preventing outliers from "
                                        "skewing the feature scaling.",
                        category="scaling",
                        metadata={"column": col, "action": "robust"}
                    )
                )
            else:
                recs.append(
                    Recommendation(
                        label=f"Standard Scale: {col}",
                        confidence_score=0.92,
                        reason=f"Column '{col}' is normally distributed without extreme outliers.",
                        why_explanation="Standard scaling centers variables to zero mean and unit variance. "
                                        "This is highly recommended for gradient-descent models (Neural Networks, Custom SVM, Custom kNN).",
                        category="scaling",
                        metadata={"column": col, "action": "standard"}
                    )
                )

        # Sort recommendations
        recs.sort(key=lambda r: r.confidence_score, reverse=True)
        return recs
