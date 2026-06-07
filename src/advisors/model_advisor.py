"""Smart Model Advisor module.

Recommends ML models based on dataset characteristics including
size, problem type, feature types, class balance, and complexity.
"""

import logging
from typing import Any

from src.advisors.base import BaseAdvisor, Recommendation
from config.settings import SETTINGS

logger = logging.getLogger(__name__)


class ModelAdvisor(BaseAdvisor):
    """Recommends ML models based on dataset profile."""

    def recommend(self, profile: dict[str, Any]) -> list[Recommendation]:
        """Generate model recommendations.

        Args:
            profile: Dictionary produced by DatasetProfiler.compute_profile().

        Returns:
            List of Recommendation objects sorted by confidence_score descending.
        """
        logger.info("Generating model recommendations...")
        recs = []

        rows = profile.get("shape", (100, 2))[0]
        # We can assume problem type is stored or derived.
        # Since the session state target/problem_type drives this, we'll recommend for both types
        # depending on if class balance exists (indicating classification).
        class_balance = profile.get("class_balance", {})
        is_classification = len(class_balance) > 0
        
        categorical_cols = profile.get("categorical_columns", [])
        has_categorical = len(categorical_cols) > 0

        # Classification recommendations
        if is_classification:
            # 1. XGBoost
            recs.append(
                Recommendation(
                    label="XGBoost Classifier",
                    confidence_score=0.95 if rows >= SETTINGS.SMALL_DATASET_ROWS else 0.82,
                    reason="State-of-the-art gradient boosting for tabular data, handles non-linearities and missing values.",
                    why_explanation="XGBoost builds sequential decision trees using gradient descent on the loss function. "
                                    "It handles tabular data exceptionally well and includes regularization to prevent overfitting.",
                    category="Classification",
                    metadata={"model_name": "XGBoost"}
                )
            )

            # 2. LightGBM
            recs.append(
                Recommendation(
                    label="LightGBM Classifier",
                    confidence_score=0.96 if rows >= SETTINGS.MEDIUM_DATASET_ROWS else 0.78,
                    reason="Fast leaf-wise tree growth, highly suited for medium to large datasets.",
                    why_explanation="LightGBM grows trees leaf-wise rather than level-wise, optimizing for speed and memory. "
                                    "It is highly recommended for larger datasets but might overfit on very small tables.",
                    category="Classification",
                    metadata={"model_name": "LightGBM"}
                )
            )

            # 3. CatBoost
            recs.append(
                Recommendation(
                    label="CatBoost Classifier",
                    confidence_score=0.93 if has_categorical else 0.85,
                    reason="Specialized handling of categorical features and symmetric tree structure.",
                    why_explanation="CatBoost uses ordered boosting and symmetric trees to reduce overfitting. "
                                    "It has native, highly optimized support for categorical column indices.",
                    category="Classification",
                    metadata={"model_name": "CatBoost"}
                )
            )

            # 4. Random Forest
            recs.append(
                Recommendation(
                    label="Random Forest Classifier",
                    confidence_score=0.90 if rows < SETTINGS.MEDIUM_DATASET_ROWS else 0.85,
                    reason="Robust bagging ensemble, less prone to overfitting on small/medium datasets.",
                    why_explanation="Random Forest averages predictions from independent decision trees trained on "
                                    "subsets of the data. It requires minimal hyperparameter tuning to get robust baselines.",
                    category="Classification",
                    metadata={"model_name": "Random Forest"}
                )
            )

            # 5. Logistic Regression
            recs.append(
                Recommendation(
                    label="Logistic Regression",
                    confidence_score=0.82 if rows < SETTINGS.SMALL_DATASET_ROWS else 0.70,
                    reason="Simple, interpretable linear baseline. Highly efficient.",
                    why_explanation="Logistic Regression models linear relationships using the sigmoid function. "
                                    "It serves as a fast baseline before moving to non-linear model families.",
                    category="Classification",
                    metadata={"model_name": "Logistic Regression"}
                )
            )

            # 6. Naive Bayes
            recs.append(
                Recommendation(
                    label="Naive Bayes Classifier",
                    confidence_score=0.75 if rows < SETTINGS.SMALL_DATASET_ROWS else 0.60,
                    reason="Fast probabilistic classifier assuming feature independence.",
                    why_explanation="Naive Bayes is highly efficient and performs well with text-like representations, "
                                    "assuming conditional independence between features.",
                    category="Classification",
                    metadata={"model_name": "Naive Bayes"}
                )
            )

            # 7. Custom SVM (implemented from scratch)
            recs.append(
                Recommendation(
                    label="Custom SVM (From Scratch)",
                    confidence_score=0.85 if rows < SETTINGS.SMALL_DATASET_ROWS else 0.55,
                    reason=f"From-scratch linear/RBF SVM. Works best on small datasets (current size: {rows} rows).",
                    why_explanation="Our custom Support Vector Machine optimizes the margin using hinge loss and "
                                    "gradient descent. Since it is implemented from scratch in pure Python, it is ideal "
                                    "for smaller datasets, as it runs slower on large scale datasets than C-based library packages.",
                    category="Classification",
                    metadata={"model_name": "Custom SVM"}
                )
            )

            # 8. Custom KNN (implemented from scratch)
            recs.append(
                Recommendation(
                    label="Custom KNN (From Scratch)",
                    confidence_score=0.86 if rows < SETTINGS.SMALL_DATASET_ROWS else 0.58,
                    reason=f"From-scratch k-Nearest Neighbours. Works best on small datasets (current size: {rows} rows).",
                    why_explanation="Our custom kNN computes pairwise distances directly using NumPy. It handles distance-based "
                                    "neighborhood classification with configurable metric. As a non-parametric model, "
                                    "its prediction time scales with dataset size, making it most suitable for small datasets.",
                    category="Classification",
                    metadata={"model_name": "Custom KNN"}
                )
            )
        
        else:
            # Regression models
            recs.append(
                Recommendation(
                    label="XGBoost Regressor",
                    confidence_score=0.94 if rows >= SETTINGS.SMALL_DATASET_ROWS else 0.80,
                    reason="State-of-the-art gradient boosting regressor for non-linear continuous mapping.",
                    why_explanation="XGBoost Regressor builds tree ensembles to map complex continuous functions.",
                    category="Regression",
                    metadata={"model_name": "XGBoost"}
                )
            )

            recs.append(
                Recommendation(
                    label="LightGBM Regressor",
                    confidence_score=0.95 if rows >= SETTINGS.MEDIUM_DATASET_ROWS else 0.75,
                    reason="Highly efficient boosting regressor for larger scale continuous targets.",
                    why_explanation="LightGBM Regressor optimizes tree parameters leaf-wise for fast mean squared error reduction.",
                    category="Regression",
                    metadata={"model_name": "LightGBM"}
                )
            )

            recs.append(
                Recommendation(
                    label="CatBoost Regressor",
                    confidence_score=0.92 if has_categorical else 0.84,
                    reason="Excellent tabular regressor with symmetric tree constraints.",
                    why_explanation="CatBoost Regressor trains symmetric decision trees that provide stable forecasts.",
                    category="Regression",
                    metadata={"model_name": "CatBoost"}
                )
            )

            recs.append(
                Recommendation(
                    label="Random Forest Regressor",
                    confidence_score=0.88 if rows < SETTINGS.MEDIUM_DATASET_ROWS else 0.82,
                    reason="Averages multiple regression trees to prevent variance/overfitting.",
                    why_explanation="Random Forest Regressor builds multiple decorrelated trees, averaging predictions.",
                    category="Regression",
                    metadata={"model_name": "Random Forest"}
                )
            )

            recs.append(
                Recommendation(
                    label="Linear Regression",
                    confidence_score=0.80 if rows < SETTINGS.SMALL_DATASET_ROWS else 0.70,
                    reason="Simplest baseline regression model. Assumes linear relationship.",
                    why_explanation="Linear Regression provides maximum interpretability of feature coefficients.",
                    category="Regression",
                    metadata={"model_name": "Linear Regression"}
                )
            )

        # Sort recommendations
        recs.sort(key=lambda r: r.confidence_score, reverse=True)
        return recs
