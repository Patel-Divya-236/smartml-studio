"""Split-then-fit entry point for the preprocessing stage.

This lived in `pages/4_preprocessing.py` while Streamlit was the only front end. It is
domain logic, not view code, so the React/FastAPI migration moves it here. The one
behavioural change is that the fitted LabelEncoder is *returned* rather than pushed into
`st.session_state`, which is what makes the function usable from a request handler.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.settings import SETTINGS
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


def split_and_preprocess(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float,
    impute_config: dict,
    encode_config: dict,
    scale_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder | None, PreprocessingPipeline]:
    """Split the raw dataset, then fit the preprocessing pipeline on the training rows.

    The split happens *before* any transformer is fitted. Fitting imputers, scalers, or
    encoders on the full dataset would let test-set statistics influence the training
    representation and inflate every metric reported downstream.

    Returns:
        (X_train, X_test, y_train, y_test, label_encoder, fitted_pipeline). The pipeline
        comes back fitted so new rows can be transformed later for inference.
    """
    logger.info("Splitting dataset (test_size=%.2f) before fitting any transformer.", test_size)

    working_df = df.copy()

    # ── 1. Rows with a missing target cannot be trained or scored on ──
    missing_target = working_df[target_col].isnull().sum()
    if missing_target > 0:
        logger.info("Dropping %d rows with a missing target value.", missing_target)
        working_df = working_df.dropna(subset=[target_col])

    y = working_df[target_col].copy()
    X = working_df.drop(columns=[target_col])

    # ── 2. Encode the target labels ───────────────────────────────────
    # Fitted on the full target on purpose: a LabelEncoder is a label vocabulary, not a
    # statistic learned from feature values, so it carries no information about the
    # feature distribution. Fitting on train alone would raise on any class that happens
    # to appear only in the test split.
    label_encoder: LabelEncoder | None = None
    if problem_type == "Classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)

    # ── 3. Split ──────────────────────────────────────────────────────
    stratify = None
    if problem_type == "Classification":
        counts = y.value_counts()
        # Stratifying needs at least two classes with at least two members each.
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = y
        else:
            logger.info(
                "Stratification skipped: %d classes, smallest has %d rows.",
                len(counts),
                counts.min(),
            )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SETTINGS.DEFAULT_RANDOM_STATE,
        stratify=stratify,
    )

    # ── 4. Fit on train only, then replay onto test ───────────────────
    pipeline = PreprocessingPipeline(target_col, impute_config, encode_config, scale_config)
    X_train = pipeline.fit_transform(X_train_raw, y_train)
    X_test = pipeline.transform(X_test_raw)

    logger.info("Preprocessing complete. Train %s, test %s.", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, label_encoder, pipeline
