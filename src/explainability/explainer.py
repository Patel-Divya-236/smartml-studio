"""Explainability module.

Provides SHAP-based explanations including summary plots,
force plots, waterfall plots, and model evaluation curves.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


import shap

class ModelExplainer:
    """Generates SHAP-based explanations for trained models."""

    def __init__(self) -> None:
        """Initialise the ModelExplainer."""
        logger.info("ModelExplainer initialised.")

    def explain(self, model: Any, X_train: Any, X_test: Any,
                feature_names: list[str] | None = None) -> dict[str, Any]:
        """Generate SHAP explanations for the given model.

        Supports Tree, Linear, and Kernel fallback explainers.
        """
        logger.info("Computing SHAP values for model...")
        
        # Check background training data size and summarize to speed up Kernel SHAP
        if len(X_train) > 20:
            # Summarize training data using shap.kmeans or sampling
            try:
                background = shap.kmeans(X_train, 5)
            except Exception:
                background = X_train[:5]
        else:
            background = X_train

        explainer_type = "Kernel"
        explainer = None
        shap_values = None
        base_value = 0.0

        # Try to use fast Tree/Linear explainers first
        try:
            model_class_name = model.__class__.__name__
            if "Tree" in model_class_name or "Forest" in model_class_name or "XGB" in model_class_name or "LGBM" in model_class_name or "CatBoost" in model_class_name:
                explainer = shap.TreeExplainer(model)
                explainer_type = "Tree"
                shap_values = explainer(X_test)
            elif "Linear" in model_class_name or "Logistic" in model_class_name:
                explainer = shap.LinearExplainer(model, background)
                explainer_type = "Linear"
                shap_values = explainer(X_test)
            else:
                # Fallback to Kernel/Generic explainer
                # To be fast, we evaluate on a maximum of 10 test samples if it's Kernel SHAP
                test_subset = X_test[:10] if len(X_test) > 10 else X_test
                explainer = shap.KernelExplainer(model.predict, background)
                explainer_type = "Kernel"
                shap_values = explainer.shap_values(test_subset)
        except Exception as e:
            logger.warning("Failed standard SHAP explainer initialization: %s. Using basic Kernel fallback.", str(e))
            try:
                test_subset = X_test[:5] if len(X_test) > 5 else X_test
                explainer = shap.KernelExplainer(model.predict, background)
                explainer_type = "Kernel"
                shap_values = explainer.shap_values(test_subset)
            except Exception as inner_e:
                logger.error("All SHAP explainers failed: %s", str(inner_e))
                raise inner_e

        # Extract base value depending on structure
        try:
            if hasattr(explainer, "expected_value"):
                base_value = explainer.expected_value
            elif hasattr(shap_values, "base_values"):
                base_value = shap_values.base_values
        except Exception:
            pass

        return {
            "shap_values": shap_values,
            "base_value": base_value,
            "explainer_type": explainer_type,
            "is_subset": explainer_type == "Kernel" and len(X_test) > len(shap_values)
        }

