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
            "output_space": infer_output_space(model, explainer_type),
            "is_subset": explainer_type == "Kernel" and len(X_test) > len(shap_values)
        }


def extract_sample_contributions(shap_values: Any, sample_idx: int) -> Any:
    """Return the 1-D SHAP contribution vector for one sample.

    SHAP returns several shapes depending on the explainer and the task: an Explanation
    object, a list of per-class arrays for multiclass, a 3-D array indexed by class, or
    a plain 2-D array. This normalises all of them to one row so callers do not each
    reimplement the same branching.

    Args:
        shap_values: Whatever ``ModelExplainer.explain`` placed in ``shap_values``.
        sample_idx: Index of the sample within the explained subset.

    Returns:
        A 1-D numpy array of per-feature contributions, or None if it cannot be derived.
    """
    import numpy as np

    values = shap_values

    # shap.Explanation objects carry the array on .values
    if hasattr(values, "values"):
        values = values.values

    # Multiclass explainers return one array per class; class 0 is representative
    if isinstance(values, list):
        if not values:
            return None
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if sample_idx >= values.shape[0]:
            return None
        return values[sample_idx]
    if values.ndim == 3:
        # (samples, features, classes) — take the first class
        if sample_idx >= values.shape[0]:
            return None
        return values[sample_idx, :, 0]

    logger.warning("Unrecognised SHAP value shape: %s", getattr(values, "shape", type(values)))
    return None


def infer_output_space(model: Any, explainer_type: str) -> str:
    """Describe the units SHAP contributions are expressed in, for this model.

    This is not cosmetic. SHAP contributions are only interpretable as probability for
    some model/explainer combinations, and a reader told "+0.31" means "+31 percentage
    points" is being misled whenever the space is log-odds. The narration layer refuses
    to state a unit unless this function is confident of one.

    Note the Kernel branch above wraps ``model.predict`` rather than ``predict_proba``,
    so for a classifier its contributions are in units of the *class index* -- not a
    probability, and not a log-odd.
    """
    name = model.__class__.__name__
    is_classifier = (
        getattr(model, "_estimator_type", None) == "classifier"
        or hasattr(model, "predict_proba")
        or "Classifier" in name
        or "Logistic" in name
    )

    if not is_classifier:
        return "target units"

    if explainer_type == "Kernel":
        # KernelExplainer was built on model.predict, so it explains the class index.
        return "predicted class index (NOT probability)"
    if explainer_type == "Linear":
        return "log-odds (NOT probability)"
    if explainer_type == "Tree":
        # sklearn forests expose probability as their raw output; boosted ensembles
        # expose the additive log-odds margin.
        if "Forest" in name or "ExtraTrees" in name or "DecisionTree" in name:
            return "probability"
        return "log-odds (NOT probability)"

    return "unspecified model output units"


def aggregate_global_importance(
    shap_values: Any,
    feature_names: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]] | None:
    """Rank features by mean absolute SHAP value across all explained samples.

    This is the number the beeswarm summary plot encodes as horizontal spread. Returns
    ``None`` when the SHAP output cannot be reduced to one value per feature, so callers
    can skip narration and leave the plot untouched.
    """
    import numpy as np

    values = shap_values
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if not values:
            return None
        # Multiclass: average magnitude across classes.
        try:
            values = np.mean([np.abs(np.asarray(v, dtype=float)) for v in values], axis=0)
        except (TypeError, ValueError):
            return None

    try:
        arr = np.abs(np.asarray(values, dtype=float))
    except (TypeError, ValueError):
        return None

    if arr.ndim == 3:
        # (samples, features, classes) -> collapse the class axis, then samples.
        arr = arr.mean(axis=2)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None

    means = arr.mean(axis=0)
    n = min(len(feature_names), means.shape[0])
    if n == 0:
        return None

    ranked = sorted(
        ((str(feature_names[i]), float(means[i])) for i in range(n)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[:top_n]


def select_explanation_slice(
    shap_values: Any,
    sample_idx: int,
    class_idx: int | None = None,
    feature_names: list[str] | None = None,
    feature_values: Any = None,
    base_value: Any = None,
) -> Any:
    """Reduce any SHAP output to a single-sample Explanation the waterfall can plot.

    ``shap.plots.waterfall`` accepts exactly one thing: an Explanation whose values are
    one-dimensional. Every other SHAP shape raises. Those shapes are not exotic --
    a binary tree model returns (samples, features, 2), multiclass returns
    (samples, features, n_classes), and the Kernel fallback returns a bare ndarray with
    no Explanation wrapper at all -- so the caller cannot simply index and hope.

    Returns None when no single-sample slice can be formed, letting the caller fall back
    to a plain bar chart rather than surfacing an error.
    """
    import numpy as np

    values = shap_values
    base = base_value

    # Unwrap an Explanation, keeping its own base values and data when present.
    if hasattr(values, "values"):
        if base is None:
            base = getattr(values, "base_values", None)
        if feature_values is None:
            feature_values = getattr(values, "data", None)
        if feature_names is None:
            feature_names = getattr(values, "feature_names", None)
        values = values.values

    # Multiclass explainers may hand back one array per class.
    if isinstance(values, list):
        if not values:
            return None
        pick = class_idx if class_idx is not None and class_idx < len(values) else 0
        values = values[pick]
        if isinstance(base, (list, tuple)) and len(base) > pick:
            base = base[pick]

    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None

    if arr.ndim == 3:                      # (samples, features, classes)
        if sample_idx >= arr.shape[0]:
            return None
        pick = class_idx if class_idx is not None and class_idx < arr.shape[2] else 0
        row = arr[sample_idx, :, pick]
        base = _base_for_class(base, pick, sample_idx)
    elif arr.ndim == 2:                    # (samples, features)
        if sample_idx >= arr.shape[0]:
            return None
        row = arr[sample_idx]
        base = _base_for_class(base, None, sample_idx)
    elif arr.ndim == 1:                    # already one sample
        row = arr
        base = _base_for_class(base, None, sample_idx)
    else:
        return None

    data = None
    if feature_values is not None:
        try:
            fv = np.asarray(feature_values, dtype=float)
            data = fv[sample_idx] if fv.ndim == 2 and sample_idx < fv.shape[0] else fv
            if data is not None and np.asarray(data).ndim != 1:
                data = None
            elif data is not None and len(data) != len(row):
                data = None
        except (TypeError, ValueError, IndexError):
            data = None

    names = list(feature_names)[: len(row)] if feature_names is not None else None

    return shap.Explanation(
        values=row,
        base_values=float(base) if base is not None else 0.0,
        data=data,
        feature_names=names,
    )


def _base_for_class(base_value: Any, class_idx: int | None, sample_idx: int) -> float | None:
    """Reduce a base value of any shape to the scalar this one slice needs."""
    import numpy as np

    if base_value is None:
        return None
    try:
        arr = np.asarray(base_value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        # Either one entry per class, or one per sample.
        if class_idx is not None and class_idx < arr.size:
            return float(arr[class_idx])
        if sample_idx < arr.size:
            return float(arr[sample_idx])
        return float(arr[0])
    if arr.ndim == 2 and sample_idx < arr.shape[0]:
        row = arr[sample_idx]
        pick = class_idx if class_idx is not None and class_idx < row.size else 0
        return float(row[pick])
    return float(arr.ravel()[0])
