"""
explainability/utils.py - Utility Functions for Explainable AI
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

Provides schema validation, data transformation, pipeline loading, and formatting
helpers for SHAP, LIME, stability analysis, and the future FastAPI backend.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = "models"
DEFAULT_DATASET_PATH = os.path.join("dataset", "healthcare-dataset-stroke-data.csv")

# Expected raw feature names in canonical order
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

EXPECTED_RAW_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

VALID_CATEGORIES = {
    "gender": ["Male", "Female", "Other"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"]
}


def validate_patient_dict(patient_data: dict) -> dict:
    """
    Validates a raw patient record dictionary.
    Ensures all 10 expected features are present with appropriate types.
    Ignores non-predictive keys such as 'id' if provided.
    """
    if not isinstance(patient_data, dict):
        raise TypeError(f"Patient data must be a dictionary, got {type(patient_data).__name__}")

    # Check for missing required features
    missing_fields = [f for f in EXPECTED_RAW_FEATURES if f not in patient_data]
    if missing_fields:
        raise ValueError(f"Missing required patient fields: {missing_fields}")

    cleaned = {}
    for feature in EXPECTED_RAW_FEATURES:
        val = patient_data[feature]
        if feature in NUMERICAL_FEATURES:
            try:
                cleaned[feature] = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be numeric, got: {val}")
        else:
            cleaned[feature] = str(val).strip()

    return cleaned


def patient_dict_to_dataframe(patient_data: dict) -> pd.DataFrame:
    """
    Validates and converts a patient dictionary to a 1-row pandas DataFrame
    matching the training schema.
    """
    cleaned = validate_patient_dict(patient_data)
    df = pd.DataFrame([cleaned], columns=EXPECTED_RAW_FEATURES)
    return df


def load_saved_pipeline(model_path: str):
    """
    Loads a serialized joblib pipeline or model artifact from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)


def load_default_pipeline(model_name: str = "tuned_xgboost"):
    """
    Convenience loader for standard project models from the models/ directory:
    - 'tuned_xgboost' (default)
    - 'xgboost' (base XGBoost)
    - 'random_forest'
    - 'logistic_regression'
    - 'stacking_ensemble'
    """
    mapping = {
        "tuned_xgboost": "tuned_xgboost_pipeline.joblib",
        "xgboost": "xgboost_pipeline.joblib",
        "random_forest": "random_forest_pipeline.joblib",
        "logistic_regression": "logistic_regression_pipeline.joblib",
        "stacking_ensemble": "stacking_ensemble_pipeline.joblib"
    }

    if model_name not in mapping:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(mapping.keys())}")

    path = os.path.join(MODELS_DIR, mapping[model_name])
    return load_saved_pipeline(path)


def load_preprocessor(preprocessor_path: str = os.path.join(MODELS_DIR, "preprocessor.joblib")):
    """
    Loads the standalone fitted ColumnTransformer from models/preprocessor.joblib.
    """
    return load_saved_pipeline(preprocessor_path)


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """
    Extracts the output feature names from a fitted ColumnTransformer.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out().tolist()
    raise AttributeError("Preprocessor does not have 'get_feature_names_out'. Ensure it is fitted.")


def get_training_background(n_samples: int = 50, random_state: int = 42, preprocessed: bool = False):
    """
    Loads a representative sample of training data for model-agnostic explainers (KernelSHAP / LIME).
    Uses the stratified training partition to avoid test leakage.
    """
    from sklearn.model_selection import train_test_split
    data = pd.read_csv(DEFAULT_DATASET_PATH)
    if "id" in data.columns:
        data = data.drop(columns=["id"])
    X = data.drop(columns=["stroke"])
    y = data["stroke"]

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    sample_df = X_train.sample(n=min(n_samples, len(X_train)), random_state=random_state)

    if preprocessed:
        preprocessor = load_preprocessor()
        return preprocessor.transform(sample_df)
    return sample_df


def format_explanation_for_json(
    feature_names: list,
    contributions: list | np.ndarray,
    base_value: float = None,
    prediction_prob: float = None
) -> dict:
    """
    Formats feature contributions into a sorted, JSON-serializable list of dictionaries
    for easy consumption by REST APIs, frontends, or reports.
    """
    contributions = np.asarray(contributions, dtype=float).flatten()

    if len(feature_names) != len(contributions):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} feature names vs {len(contributions)} contributions"
        )

    items = []
    for feat, val in zip(feature_names, contributions):
        direction = "increases_risk" if val > 0 else "decreases_risk" if val < 0 else "neutral"
        items.append({
            "feature": str(feat),
            "contribution": float(round(val, 6)),
            "absolute_contribution": float(round(abs(val), 6)),
            "direction": direction
        })

    # Sort descending by absolute contribution
    items.sort(key=lambda x: x["absolute_contribution"], reverse=True)

    result = {
        "ranked_features": items,
        "base_value": float(round(base_value, 6)) if base_value is not None else None,
        "predicted_probability": float(round(prediction_prob, 6)) if prediction_prob is not None else None
    }
    return result


def plot_feature_contributions(
    ranked_features: list,
    title: str = "Feature Risk Contributions",
    top_n: int = 10,
    save_path: str = None
):
    """
    Generates a horizontal bar chart of top positive and negative feature contributions.
    """
    top_items = ranked_features[:top_n]
    # Reverse for top-down display in barh
    top_items = top_items[::-1]

    names = [item["feature"] for item in top_items]
    values = [item["contribution"] for item in top_items]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in values]

    plt.figure(figsize=(9, 6))
    plt.barh(names, values, color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.xlabel("Explanation Contribution Value")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
