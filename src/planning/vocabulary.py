"""The closed set of choices the planner is allowed to make.

The prompt, the response parser and the validator all read these tables, so the set of
techniques the LLM is offered, the set it is parsed against and the set the pipeline can
actually execute cannot drift apart. A technique outside them is not an unusual choice --
it is an action no code path can run, and the validator rejects it.

Canonical keys are the same action strings the rule-based advisor writes into
`metadata.action`, which is what the Preprocessing page maps onto its selectors
(`IMPUTER_BY_ACTION`, `ENCODER_BY_ACTION`, `SCALER_BY_ACTION` in PreprocessingStep.tsx).
Using the same keys is what lets an LLM recommendation and a rule recommendation flow
through one UI unchanged.
"""

import re

from src.models.model_trainer import SUPPORTED_MODELS

# ── Preprocessing ───────────────────────────────────────────────────────

# Technique -> the recommendation category it is filed under. "drop" is filed under
# imputation because the pipeline reads dropped columns from `impute_config` alone
# (PreprocessingPipeline._drop_configured_columns); filed anywhere else it would be shown
# and silently ignored.
PREPROCESSING_CATEGORY: dict[str, str] = {
    "mean": "imputation",
    "median": "imputation",
    "mode": "imputation",
    "knn": "imputation",
    "drop": "imputation",
    "onehot": "encoding",
    "ordinal": "encoding",
    "standard": "scaling",
    "minmax": "scaling",
    "robust": "scaling",
    "log1p": "scaling",
}

IMPUTATION = tuple(t for t, c in PREPROCESSING_CATEGORY.items() if c == "imputation")
ENCODING = tuple(t for t, c in PREPROCESSING_CATEGORY.items() if c == "encoding")
SCALING = tuple(t for t, c in PREPROCESSING_CATEGORY.items() if c == "scaling")

# Imputers that compute an average or a distance, and so need numbers.
NUMERIC_ONLY_IMPUTERS = frozenset({"mean", "median", "knn"})

# Card titles, worded exactly as the rule-based advisor words them.
PREPROCESSING_LABELS: dict[str, str] = {
    "mean": "Mean Imputer for: {column}",
    "median": "Median Imputer for: {column}",
    "mode": "Mode (Most Frequent) Imputer for: {column}",
    "knn": "KNN Imputer for: {column}",
    "drop": "Drop Column: {column}",
    "onehot": "One-Hot Encode: {column}",
    "ordinal": "Ordinal / Label Encode: {column}",
    "standard": "Standard Scale: {column}",
    "minmax": "MinMax Scale: {column}",
    "robust": "Robust Scale: {column}",
    "log1p": "Log1p Transform: {column}",
}

# What each technique does, for the "Why?" panel. The planner's own reason is specific to
# this dataset; this is the general mechanism a beginner needs alongside it.
TECHNIQUE_EXPLANATIONS: dict[str, str] = {
    "mean": "Fills each missing value with the column's average. Fast and simple, but "
            "pulled towards extreme values when the column has outliers.",
    "median": "Fills each missing value with the column's middle value. Unlike the mean, "
              "it is not dragged around by outliers or a long tail.",
    "mode": "Fills each missing value with the most common value in the column. Works for "
            "text categories as well as numbers.",
    "knn": "Fills each missing value from the most similar complete rows, so the filled "
           "value respects how this column relates to the others.",
    "drop": "Removes the column entirely. Used when it is mostly missing, or when it is an "
            "identifier or free text that would teach the model nothing general.",
    "onehot": "Turns each category into its own yes/no column. No artificial order is "
              "implied, but every category adds a column.",
    "ordinal": "Replaces each category with a number. Compact however many categories "
               "there are, but implies an order the categories may not have.",
    "standard": "Rescales the column to mean 0 and standard deviation 1, which distance- "
                "and gradient-based models need to treat features on equal terms.",
    "minmax": "Rescales the column into the range 0 to 1. Keeps the shape of the "
              "distribution, but one extreme value squeezes everything else together.",
    "robust": "Rescales using the median and interquartile range, so outliers do not "
              "decide the scale the way they do with standard scaling.",
    "log1p": "Applies log(1 + x), which compresses a long right tail so a few very large "
             "values stop dominating the column.",
}

# ── Feature engineering ─────────────────────────────────────────────────

FEATURE_ENGINEERING = ("low_variance", "polynomial", "pca", "select_k_best")

FEATURE_EXPLANATIONS: dict[str, str] = {
    "low_variance": "Removes features that barely change from row to row. A column that "
                    "is almost constant cannot help tell rows apart.",
    "polynomial": "Adds products and powers of the numeric features, so models that only "
                  "draw straight lines can capture curves and interactions.",
    "pca": "Compresses correlated numeric features into fewer uncorrelated components "
           "that keep most of the variation, at the cost of readable feature names.",
    "select_k_best": "Keeps only the k features most statistically related to the "
                     "target, scored on the training rows only so the test set cannot leak in.",
}

# ── Models ──────────────────────────────────────────────────────────────

MODEL_EXPLANATIONS: dict[str, str] = {
    "XGBoost": "Builds decision trees one after another, each correcting the last one's "
               "errors, with regularisation to limit overfitting. Strong on tabular data.",
    "LightGBM": "Gradient-boosted trees grown leaf by leaf, which makes it fast and "
                "memory-light on larger tables; it can overfit very small ones.",
    "CatBoost": "Gradient-boosted symmetric trees with careful handling of categorical "
                "features, which tends to overfit less with default settings.",
    "Random Forest": "Averages many decision trees trained on random slices of the data. "
                     "Robust, needs little tuning, and copes with mixed feature types.",
    "Logistic Regression": "Fits a weighted sum of the features to the odds of each class. "
                           "Fast, interpretable, and a strong baseline when classes "
                           "separate along straight-line boundaries.",
    "Linear Regression": "Fits a weighted sum of the features to the target. Fast and "
                         "interpretable, and the baseline every other regressor must beat.",
    "Naive Bayes": "Applies Bayes' rule assuming the features are independent given the "
                   "class. Very fast and works with little data, though the assumption "
                   "rarely holds exactly.",
    "SVM": "Finds the boundary with the widest margin between classes, using a kernel to "
           "draw curved boundaries. Strong on small and medium data, slow on large data.",
    "KNN": "Predicts from the most similar training rows. No training step, but every "
           "prediction compares against the whole training set, and it needs scaled features.",
}


def models_for(problem_type: str | None) -> tuple[str, ...]:
    """Models the trainer can build for this task type.

    Read from the trainer itself, so this list can never offer a model the training step
    would refuse. Time Series shares the continuous-target branch, as everywhere else.
    """
    key = "Classification" if problem_type == "Classification" else "Regression"
    return tuple(SUPPORTED_MODELS[key])


def model_label(name: str, problem_type: str | None) -> str:
    """Card title for a model, worded as the rule-based advisor words it."""
    if name.endswith("Regression"):
        return name
    return f"{name} {'Classifier' if problem_type == 'Classification' else 'Regressor'}"


# ── Normalisation ───────────────────────────────────────────────────────
#
# The prompt asks for these exact keys, and JSON mode makes that likely but not certain:
# a model asked for "standard" may still write "StandardScaler", the name it has seen a
# thousand times. Mapping well-known spellings onto the canonical key is deterministic and
# loses nothing; rejecting "StandardScaler" as unknown would throw away a correct answer
# over its spelling. Anything not listed here is still rejected.

_TECHNIQUE_ALIASES: dict[str, str] = {}
for _canonical, _spellings in {
    "mean": ["mean", "meanimputer", "meanimputation", "simpleimputermean"],
    "median": ["median", "medianimputer", "medianimputation", "simpleimputermedian"],
    "mode": ["mode", "mostfrequent", "modeimputer", "modeimputation",
             "mostfrequentimputer", "simpleimputermostfrequent"],
    "knn": ["knn", "knnimputer", "knnimputation", "knearestneighborsimputer"],
    "drop": ["drop", "dropcolumn", "dropfeature", "removecolumn", "remove"],
    "onehot": ["onehot", "onehotencoder", "onehotencoding", "dummies", "getdummies",
               "dummyencoding"],
    "ordinal": ["ordinal", "ordinalencoder", "ordinalencoding", "label", "labelencoder",
                "labelencoding"],
    "standard": ["standard", "standardscaler", "standardscaling", "standardize",
                 "standardise", "standardization", "standardisation", "zscore"],
    "minmax": ["minmax", "minmaxscaler", "minmaxscaling"],
    "robust": ["robust", "robustscaler", "robustscaling"],
    "log1p": ["log1p", "log1ptransform", "logtransform", "log", "logtransformation"],
    "low_variance": ["lowvariance", "variancethreshold", "lowvariancefilter",
                     "lowvarianceremoval", "removelowvariance"],
    "polynomial": ["polynomial", "polynomialfeatures", "poly", "polynomialexpansion"],
    "pca": ["pca", "principalcomponentanalysis", "principalcomponents"],
    "select_k_best": ["selectkbest", "kbest", "selectk", "univariateselection"],
}.items():
    for _spelling in _spellings:
        _TECHNIQUE_ALIASES[_spelling] = _canonical

_MODEL_ALIASES: dict[str, str] = {
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "logreg": "Logistic Regression",
    "linreg": "Linear Regression",
    "nb": "Naive Bayes",
    "gaussiannb": "Naive Bayes",
    "svc": "SVM",
    "svr": "SVM",
    "supportvectormachine": "SVM",
    "kneighbors": "KNN",
    "knearestneighbors": "KNN",
    "kneighbours": "KNN",
}


def _squash(value: object) -> str:
    """Lower-case and strip everything but letters and digits."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalise_technique(value: object) -> str | None:
    """Map a technique name onto its canonical key, or None when it is not one."""
    return _TECHNIQUE_ALIASES.get(_squash(value))


def normalise_model(value: object, allowed: tuple[str, ...]) -> str | None:
    """Map a model name onto one of `allowed`, or None when it is none of them.

    Class-name suffixes are ignored ("XGBClassifier", "RandomForestRegressor"), since the
    task type -- not the model's name -- decides classifier versus regressor.
    """
    squashed = _squash(value)
    for suffix in ("classifier", "regressor", "classification", "regression"):
        if squashed.endswith(suffix) and squashed not in {"logisticregression", "linearregression"}:
            squashed = squashed[: -len(suffix)]
            break

    by_key = {_squash(name): name for name in allowed}
    if squashed in by_key:
        return by_key[squashed]
    aliased = _MODEL_ALIASES.get(squashed)
    return aliased if aliased in allowed else None
