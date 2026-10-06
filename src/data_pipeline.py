"""
data_pipeline.py - Reusable Data Preprocessing and Splitting Module
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

This module provides standardized, reusable functions for loading the stroke dataset,
configuring ColumnTransformer preprocessing, and performing stratified data splitting
without data leakage.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Canonical feature schema based on existing research scripts
NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

TARGET_COLUMN = "stroke"
ID_COLUMN = "id"
DEFAULT_DATASET_PATH = os.path.join("dataset", "healthcare-dataset-stroke-data.csv")


def load_dataset(filepath: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Load raw healthcare stroke dataset from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath)


def get_features_and_target(
    data: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN
):
    """
    Separate predictors (X) and binary target (y), dropping non-predictive identifier column.
    """
    cols_to_drop = [target_col]
    if id_col in data.columns:
        cols_to_drop.append(id_col)
    
    X = data.drop(columns=cols_to_drop)
    y = data[target_col]
    return X, y


def create_preprocessor(
    numerical_features: list = NUMERICAL_FEATURES,
    categorical_features: list = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Create standard ColumnTransformer preprocessing pipeline:
    - Numerical: median imputation + standard scaling
    - Categorical: most frequent imputation + one-hot encoding (ignore unknown)
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Perform stratified train/test split to preserve target class distribution.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def load_and_prep_data(
    filepath: str = DEFAULT_DATASET_PATH,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Convenience function that loads, extracts, and splits the data into:
    X_train, X_test, y_train, y_test, and an unfitted preprocessor ColumnTransformer.
    """
    data = load_dataset(filepath)
    X, y = get_features_and_target(data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    preprocessor = create_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor
