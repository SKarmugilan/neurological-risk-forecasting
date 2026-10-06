"""
train_and_save_models.py - Model Training and Persistence Pipeline
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

This script trains all benchmark models, the hyperparameter-tuned model, and the
stacking ensemble, saving serialized pipeline artifacts (.joblib) into models/ along
with comprehensive metadata in models/model_metadata.json.
"""

import os
import json
import datetime
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

from data_pipeline import (
    load_and_prep_data,
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    DEFAULT_DATASET_PATH
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_definitions():
    """
    Returns exact model definitions preserving the existing experimental parameters:
    - Logistic Regression: class_weight='balanced', max_iter=1000
    - Random Forest: n_estimators=300, random_state=42, class_weight='balanced'
    - Base XGBoost: n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=19.54, eval_metric='logloss', random_state=42
    - Tuned XGBoost: n_estimators=200, max_depth=4, learning_rate=0.1, subsample=1.0,
                     colsample_bytree=1.0, min_child_weight=5, scale_pos_weight=19.54,
                     eval_metric='logloss', random_state=42
    """
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    base_xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=19.54,
        eval_metric="logloss",
        random_state=42
    )

    tuned_xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=5,
        scale_pos_weight=19.54,
        eval_metric="logloss",
        random_state=42
    )

    return {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model,
        "xgboost": base_xgb_model,
        "tuned_xgboost": tuned_xgb_model
    }


def build_stacking_ensemble(tuned_xgb_model):
    """
    Constructs the StackingClassifier ensemble matching src/ensemble_model.py:
    Base estimators: Logistic Regression, Random Forest, Tuned XGBoost (each with preprocessor)
    Meta-learner: LogisticRegression(max_iter=1000)
    Cross-validation: 5-fold out-of-fold predict_proba stacking
    """
    estimators = [
        (
            "logistic",
            Pipeline([
                ("preprocessing", create_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
        ),
        (
            "random_forest",
            Pipeline([
                ("preprocessing", create_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
            ])
        ),
        (
            "xgboost",
            Pipeline([
                ("preprocessing", create_preprocessor()),
                ("model", tuned_xgb_model)
            ])
        )
    ]

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    return ensemble


def train_and_persist_pipelines():
    print("=" * 70)
    print("PHASE 2: TRAINING AND SAVING MODEL PIPELINES")
    print("=" * 70)

    # 1. Load and split data
    print("\n1. Loading and splitting dataset...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_prep_data(
        DEFAULT_DATASET_PATH,
        test_size=0.20,
        random_state=42
    )
    print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"   Stroke cases in train: {y_train.sum()} ({y_train.mean():.2%})")

    # 2. Fit and save standalone preprocessor for explainability module (Phase 3)
    print("\n2. Fitting and saving standalone Preprocessor...")
    preprocessor.fit(X_train)
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"   Saved: {preprocessor_path} ({len(feature_names)} transformed features)")

    # 3. Train and save individual model pipelines
    models = get_model_definitions()
    saved_artifacts = {
        "preprocessor": preprocessor_path
    }

    print("\n3. Training individual end-to-end model pipelines...")
    for model_key, model_instance in models.items():
        print(f"   Training {model_key}...")
        pipeline = Pipeline([
            ("preprocessing", create_preprocessor()),
            ("model", model_instance)
        ])
        pipeline.fit(X_train, y_train)

        pipeline_filename = f"{model_key}_pipeline.joblib"
        pipeline_path = os.path.join(MODELS_DIR, pipeline_filename)
        joblib.dump(pipeline, pipeline_path)
        saved_artifacts[model_key] = pipeline_path
        print(f"   Saved: {pipeline_path}")

    # 4. Train and save Stacking Ensemble
    print("\n4. Training Stacking Ensemble...")
    ensemble = build_stacking_ensemble(models["tuned_xgboost"])
    ensemble.fit(X_train, y_train)

    ensemble_path = os.path.join(MODELS_DIR, "stacking_ensemble_pipeline.joblib")
    joblib.dump(ensemble, ensemble_path)
    saved_artifacts["stacking_ensemble"] = ensemble_path
    print(f"   Saved: {ensemble_path}")

    # 5. Create and save metadata JSON
    print("\n5. Generating and saving metadata...")
    metadata = {
        "project_title": "Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting",
        "description": "Trained machine learning models and preprocessing pipelines for early neurological risk forecasting research.",
        "disclaimer": "This model is a research prototype for early neurological risk exploration and is NOT clinically validated. It must not be used as a medical diagnostic tool.",
        "training_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": {
            "filepath": DEFAULT_DATASET_PATH,
            "total_samples": len(X_train) + len(X_test),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": 0.20,
            "stratified": True,
            "random_state": 42,
            "target_column": TARGET_COLUMN,
            "target_positive_label": 1,
            "target_negative_label": 0
        },
        "feature_schema": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "total_raw_features": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
            "total_preprocessed_features": len(feature_names),
            "preprocessed_feature_names": feature_names
        },
        "tuned_xgboost_hyperparameters": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 5,
            "scale_pos_weight": 19.54,
            "eval_metric": "logloss",
            "random_state": 42
        },
        "stacking_ensemble_architecture": {
            "base_estimators": [
                "Logistic Regression (class_weight='balanced')",
                "Random Forest (n_estimators=300, class_weight='balanced')",
                "Tuned XGBoost (PR-AUC optimized, scale_pos_weight=19.54)"
            ],
            "meta_learner": "Logistic Regression (max_iter=1000)",
            "cv_folds": 5,
            "stack_method": "predict_proba"
        },
        "decision_thresholds": {
            "default_standard": 0.50,
            "recommended_balanced": 0.15,
            "recommended_screening": 0.10,
            "threshold_notes": "At default 0.50 threshold, probability collapse occurs due to extreme class imbalance (sensitivity=0.0). Threshold 0.15 is supported by experimental results providing optimal F1 (0.3600), MCC (0.3375), and balanced sensitivity (0.54) vs specificity (0.925)."
        },
        "saved_artifacts": saved_artifacts
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"   Saved: {metadata_path}")

    print("\n" + "=" * 70)
    print("TRAINING AND PERSISTENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    train_and_persist_pipelines()
