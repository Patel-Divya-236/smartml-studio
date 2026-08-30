"""Preprocessing (split-then-fit) and feature engineering."""

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_session, require
from backend.core.serialization import dataframe_to_records, to_jsonable
from backend.core.session import Session
from src.features.feature_engineering import FeatureEngineeringPipeline
from src.preprocessing.split import split_and_preprocess

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class PreprocessRequest(BaseModel):
    """Per-column transformer choices plus the split ratio."""

    test_size: float = Field(default=0.2, gt=0.05, lt=0.6)
    impute: dict[str, str] = {}
    encode: dict[str, str] = {}
    scale: dict[str, str] = {}


class FeatureRequest(BaseModel):
    """Opt-in feature engineering steps.

    Field names mirror `FeatureEngineeringPipeline`'s config keys exactly so the request
    can be passed straight through without a translation table drifting out of sync.
    """

    low_variance_active: bool = False
    low_variance_threshold: float = 0.01
    poly_active: bool = False
    poly_degree: int = 2
    poly_interaction_only: bool = False
    pca_active: bool = False
    pca_components: int = 2
    select_k_best_active: bool = False
    select_k_best_k: int = 5
    select_k_best_score_func: str = "f_test"


@router.post("/preprocess")
def preprocess(payload: PreprocessRequest, session: Session = Depends(get_session)) -> dict:
    """Split the dataset, then fit every transformer on the training rows only.

    Fitting before splitting would let test-set statistics reach the imputers, scalers
    and encoders, inflating every metric reported later. The order here is the guarantee.
    """
    df: pd.DataFrame = require(session, "dataset", "upload")
    target = session.get("target_column")
    problem_type = session.get("problem_type")
    if not target:
        raise HTTPException(status_code=409, detail="Choose a target column first.")

    try:
        X_train, X_test, y_train, y_test, label_encoder, fitted = split_and_preprocess(
            df, target, problem_type, payload.test_size,
            payload.impute, payload.encode, payload.scale,
        )
    except Exception as exc:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}") from exc

    session.reset_downstream("preprocessing_recommendations")
    session.set("test_size", payload.test_size)
    session.set("label_encoder", label_encoder)
    session.set("preprocessing_config", payload.model_dump())
    session.set("preprocessed_train", X_train)
    session.set("preprocessed_test", X_test)
    session.set("y_train", y_train)
    session.set("y_test", y_test)
    session.set("preprocessing_pipeline", fitted)
    session.set("feature_names", list(X_train.columns))

    return {
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "feature_names": [str(c) for c in X_train.columns],
        "dropped_columns": [str(c) for c in fitted.dropped_columns],
        "classes": [str(c) for c in label_encoder.classes_] if label_encoder is not None else None,
        "preview": dataframe_to_records(X_train, limit=20),
        "completed_steps": session.completed_steps(),
    }


@router.post("/features")
def features(payload: FeatureRequest, session: Session = Depends(get_session)) -> dict:
    """Apply the opt-in feature engineering steps, fitting on the training rows only.

    SelectKBest is the sharpest case: fitting it on the full dataset would choose
    features using the test set's target values, which is direct label leakage.
    """
    X_train: pd.DataFrame = require(session, "preprocessed_train", "preprocessing")
    X_test: pd.DataFrame = session.get("preprocessed_test")
    y_train = session.get("y_train")
    config = payload.model_dump()

    any_active = any(config.get(k) for k in config if k.endswith("_active"))
    if not any_active:
        # Passing through is a legitimate choice, not an error.
        out_train, out_test = X_train, X_test
        names = list(X_train.columns)
    else:
        try:
            pipeline = FeatureEngineeringPipeline(config, session.get("problem_type"))
            out_train = pipeline.fit_transform(X_train, y_train)
            out_test = pipeline.transform(X_test)
            names = list(pipeline.feature_names_out)
        except Exception as exc:
            logger.exception("Feature engineering failed")
            raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc

    session.reset_downstream("feature_config")
    session.set("feature_config", config)
    session.set("feature_engineered_train", out_train)
    session.set("feature_engineered_test", out_test)
    session.set("feature_names", [str(n) for n in names])

    return {
        "train_shape": [int(out_train.shape[0]), int(out_train.shape[1])],
        "test_shape": [int(out_test.shape[0]), int(out_test.shape[1])],
        "feature_names": [str(n) for n in names],
        "features_before": int(X_train.shape[1]),
        "features_after": int(out_train.shape[1]),
        "preview": dataframe_to_records(out_train, limit=20),
        "completed_steps": session.completed_steps(),
    }


@router.get("/state")
def pipeline_state(session: Session = Depends(get_session)) -> dict:
    """Report which steps are complete, so the sidebar rail can lock the rest."""
    train = session.get("feature_engineered_train")
    if train is None:
        train = session.get("preprocessed_train")

    return {
        "completed_steps": session.completed_steps(),
        "dataset_name": session.get("dataset_name"),
        "target_column": session.get("target_column"),
        "problem_type": session.get("problem_type"),
        "test_size": to_jsonable(session.get("test_size")),
        "feature_count": int(train.shape[1]) if train is not None else None,
        "trained_model_names": list((session.get("trained_models") or {}).keys()),
    }
