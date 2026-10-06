"""
test_explainability.py - Test Script for Explainable AI Modules
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

This script performs verification of the explainability package:
1. TreeSHAP on Tuned XGBoost
2. Model-Agnostic LIME on Tuned XGBoost
3. Live Perturbation Stability Analysis
4. Model-Agnostic KernelSHAP on the Stacking Ensemble

==============================================================================
SOFTWARE TEST ONLY:
The sample patient below is a synthetic test record used strictly to verify
code execution, array shapes, and numerical finiteness. It is NOT a clinical record.
==============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath("."))

from explainability import (
    explain_patient_tree_shap,
    explain_ensemble_kernel_shap,
    explain_patient_lime,
    evaluate_explanation_stability,
    load_default_pipeline
)

# Synthetic technical test record
SYNTHETIC_TEST_PATIENT = {
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
}


def run_explainability_tests():
    print("=" * 80)
    print("EXPLAINABILITY MODULE VERIFICATION SUITE")
    print("=" * 80)
    print("\n[NOTE] Running verification on synthetic software test record.\n")

    test_results = []
    all_passed = True

    # --------------------------------------------------------------------------
    # TEST 1: TreeSHAP on Tuned XGBoost
    # --------------------------------------------------------------------------
    print("1. Testing TreeSHAP on Tuned XGBoost Pipeline...")
    try:
        pipeline = load_default_pipeline("tuned_xgboost")
        shap_res = explain_patient_tree_shap(
            patient_data=SYNTHETIC_TEST_PATIENT,
            pipeline=pipeline,
            model_name="tuned_xgboost"
        )

        probs = shap_res["predicted_stroke_probability"]
        features = shap_res["feature_contributions"]
        shap_vec = np.asarray(shap_res["raw_shap_vector"])

        is_finite = np.all(np.isfinite(shap_vec))
        has_features = len(features) == 21
        valid_prob = 0.0 <= probs <= 1.0

        if is_finite and has_features and valid_prob:
            print(f"   [PASS] TreeSHAP computed successfully.")
            print(f"          Predicted Stroke Probability: {probs:.4f}")
            print(f"          Base Value (Expected log-odds/prob): {shap_res['base_value']:.4f}")
            print("          Top 3 Risk Factors:")
            for item in features[:3]:
                print(f"            - {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
            test_results.append(("TreeSHAP (Tuned XGBoost)", "PASS", f"Prob={probs:.4f}, Features={len(features)}"))
        else:
            print("   [FAIL] Output validation failed (non-finite or incorrect length).")
            test_results.append(("TreeSHAP (Tuned XGBoost)", "FAIL", "Invalid output shape/values"))
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] TreeSHAP exception: {e}")
        test_results.append(("TreeSHAP (Tuned XGBoost)", "FAIL", str(e)))
        all_passed = False

    # --------------------------------------------------------------------------
    # TEST 2: LIME on Tuned XGBoost
    # --------------------------------------------------------------------------
    print("\n2. Testing LIME on Tuned XGBoost Pipeline...")
    try:
        lime_res = explain_patient_lime(
            patient_data=SYNTHETIC_TEST_PATIENT,
            pipeline=pipeline,
            model_name="tuned_xgboost",
            num_features=10,
            num_samples=1000,
            random_state=42
        )

        top_lime = lime_res["top_feature_contributions"]
        weights = [f["contribution"] for f in top_lime]
        is_finite = np.all(np.isfinite(weights))
        has_features = len(top_lime) == 10

        if is_finite and has_features:
            print(f"   [PASS] LIME explanation computed successfully.")
            print(f"          Local Surrogate Model R²: {lime_res['local_score_r2']:.4f}")
            print("          Top 3 LIME Attributions:")
            for item in top_lime[:3]:
                print(f"            - {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
            test_results.append(("LIME (Tuned XGBoost)", "PASS", f"R²={lime_res['local_score_r2']:.4f}"))
        else:
            print("   [FAIL] LIME output validation failed.")
            test_results.append(("LIME (Tuned XGBoost)", "FAIL", "Invalid weights"))
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] LIME exception: {e}")
        test_results.append(("LIME (Tuned XGBoost)", "FAIL", str(e)))
        all_passed = False

    # --------------------------------------------------------------------------
    # TEST 3: Dynamic Explanation Stability Analysis
    # --------------------------------------------------------------------------
    print("\n3. Testing Explanation Stability Under Physiological Perturbations...")
    try:
        stability_res = evaluate_explanation_stability(
            patient_data=SYNTHETIC_TEST_PATIENT,
            pipeline=pipeline,
            model_name="tuned_xgboost"
        )

        avg_cos = stability_res["average_cosine_similarity"]
        avg_mad = stability_res["average_mean_absolute_shap_difference"]
        pert_details = stability_res["perturbation_details"]

        valid_cos = -1.0 <= avg_cos <= 1.0 and np.isfinite(avg_cos)
        valid_mad = avg_mad >= 0.0 and np.isfinite(avg_mad)
        has_6_perts = len(pert_details) == 6

        if valid_cos and valid_mad and has_6_perts:
            print(f"   [PASS] Stability analysis evaluated 6 perturbations dynamically.")
            print(f"          Average Cosine Similarity: {avg_cos:.4f}")
            print(f"          Average Mean Absolute SHAP Shift: {avg_mad:.6f}")
            test_results.append(("Explanation Stability", "PASS", f"Avg CosSim={avg_cos:.4f}"))
        else:
            print("   [FAIL] Stability metrics out of valid bounds.")
            test_results.append(("Explanation Stability", "FAIL", "Invalid metrics"))
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] Stability exception: {e}")
        test_results.append(("Explanation Stability", "FAIL", str(e)))
        all_passed = False

    # --------------------------------------------------------------------------
    # TEST 4: Model-Agnostic KernelSHAP on Stacking Ensemble
    # --------------------------------------------------------------------------
    print("\n4. Testing Model-Agnostic KernelSHAP on Stacking Ensemble...")
    try:
        ensemble_pipeline = load_default_pipeline("stacking_ensemble")
        ensemble_shap = explain_ensemble_kernel_shap(
            patient_data=SYNTHETIC_TEST_PATIENT,
            ensemble_pipeline=ensemble_pipeline,
            background_samples=25,
            nsamples=60
        )

        ens_prob = ensemble_shap["predicted_stroke_probability"]
        ens_features = ensemble_shap["feature_contributions"]
        ens_vec = np.asarray(ensemble_shap["raw_shap_vector"])

        is_finite = np.all(np.isfinite(ens_vec))
        has_features = len(ens_features) == 21
        valid_prob = 0.0 <= ens_prob <= 1.0

        if is_finite and has_features and valid_prob:
            print(f"   [PASS] Ensemble KernelSHAP computed successfully.")
            print(f"          Ensemble Predicted Stroke Probability: {ens_prob:.4f}")
            print("          Top 3 Ensemble Feature Contributions:")
            for item in ens_features[:3]:
                print(f"            - {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
            test_results.append(("KernelSHAP (Stacking Ensemble)", "PASS", f"Prob={ens_prob:.4f}, Features={len(ens_features)}"))
        else:
            print("   [FAIL] Ensemble KernelSHAP output validation failed.")
            test_results.append(("KernelSHAP (Stacking Ensemble)", "FAIL", "Invalid output shape/values"))
            all_passed = False
    except Exception as e:
        print(f"   [FAIL] Ensemble KernelSHAP exception: {e}")
        test_results.append(("KernelSHAP (Stacking Ensemble)", "FAIL", str(e)))
        all_passed = False

    # --------------------------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    df_summary = pd.DataFrame(test_results, columns=["Test Component", "Status", "Details"])
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
    if all_passed:
        print("RESULT: ALL EXPLAINABILITY MODULES PASSED VERIFICATION")
    else:
        print("RESULT: ONE OR MORE EXPLAINABILITY TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_explainability_tests()
    sys.exit(0 if success else 1)
