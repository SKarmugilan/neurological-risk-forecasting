"""
verify_saved_models.py - Verification Script for Saved Model Pipelines
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

This script loads each serialized .joblib artifact from models/, verifies its
deserialization, and executes predict_proba() on synthetic software-test cases
to confirm end-to-end functionality.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "models"
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

# ==============================================================================
# SOFTWARE TEST ONLY:
# Synthetic verification records used exclusively to test pipeline deserialization
# and schema validation. These are NOT clinical records or research results.
# ==============================================================================
SYNTHETIC_TEST_CASES = pd.DataFrame([
    {
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    },
    {
        "gender": "Female",
        "age": 35.0,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Govt_job",
        "Residence_type": "Rural",
        "avg_glucose_level": 82.50,
        "bmi": 24.1,
        "smoking_status": "never smoked"
    }
])


def verify_models():
    print("=" * 75)
    print("PHASE 2: VERIFICATION OF SAVED MODEL ARTIFACTS")
    print("=" * 75)

    # 1. Verify metadata file
    print(f"\n1. Verifying metadata file ({METADATA_FILE})...")
    if not os.path.exists(METADATA_FILE):
        print(f"   [FAIL] Metadata file missing: {METADATA_FILE}")
        return False
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   [PASS] Metadata loaded. Title: {metadata.get('project_title')}")
    print(f"   [INFO] Recommended Balanced Threshold: {metadata.get('decision_thresholds', {}).get('recommended_balanced')}")

    # 2. Verify Preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    print(f"\n2. Verifying standalone Preprocessor ({preprocessor_path})...")
    if not os.path.exists(preprocessor_path):
        print(f"   [FAIL] Missing file: {preprocessor_path}")
        return False
    try:
        preprocessor = joblib.load(preprocessor_path)
        transformed = preprocessor.transform(SYNTHETIC_TEST_CASES)
        print(f"   [PASS] Preprocessor loaded successfully.")
        print(f"   [INFO] Input shape: {SYNTHETIC_TEST_CASES.shape} -> Transformed shape: {transformed.shape}")
    except Exception as e:
        print(f"   [FAIL] Preprocessor error: {e}")
        return False

    # 3. Model Pipelines to Verify
    pipelines_to_test = [
        ("Logistic Regression", "logistic_regression_pipeline.joblib"),
        ("Random Forest", "random_forest_pipeline.joblib"),
        ("Base XGBoost", "xgboost_pipeline.joblib"),
        ("Tuned XGBoost", "tuned_xgboost_pipeline.joblib"),
        ("Stacking Ensemble", "stacking_ensemble_pipeline.joblib")
    ]

    results_table = []
    all_passed = True

    print("\n3. Testing End-to-End Prediction on Synthetic Test Records...")
    print("   (Note: Software test only - verifying deserialization and shape integrity)\n")

    for model_name, filename in pipelines_to_test:
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            results_table.append({
                "Model": model_name,
                "File": filename,
                "Loaded": "NO",
                "Probabilities": "N/A",
                "Status": "FAIL (File Missing)"
            })
            all_passed = False
            continue

        try:
            model = joblib.load(filepath)
            # Run prediction on raw synthetic DataFrame
            probabilities = model.predict_proba(SYNTHETIC_TEST_CASES)
            stroke_probs = probabilities[:, 1]

            # Validate probability properties
            is_valid_shape = probabilities.shape == (len(SYNTHETIC_TEST_CASES), 2)
            is_valid_range = np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            sums_to_one = np.allclose(probabilities.sum(axis=1), 1.0)

            if is_valid_shape and is_valid_range and sums_to_one:
                status = "PASS"
                prob_str = f"Case 1: {stroke_probs[0]:.4f} | Case 2: {stroke_probs[1]:.4f}"
            else:
                status = "FAIL (Invalid Probability Values)"
                prob_str = str(stroke_probs)
                all_passed = False

            results_table.append({
                "Model": model_name,
                "File": filename,
                "Loaded": "YES",
                "Probabilities": prob_str,
                "Status": status
            })

        except Exception as e:
            results_table.append({
                "Model": model_name,
                "File": filename,
                "Loaded": "ERROR",
                "Probabilities": str(e),
                "Status": "FAIL (Inference Error)"
            })
            all_passed = False

    # 4. Summary Table
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL MODEL ARTIFACTS VERIFIED SUCCESSFULLY: READY FOR INFERENCE")
    else:
        print("VERIFICATION FAILED FOR ONE OR MORE ARTIFACTS")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)
