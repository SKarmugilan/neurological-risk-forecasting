import pandas as pd
import os

# ============================================================
# MASTER RESULTS TABLE
# ============================================================

results_dir = "results"

# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

model_results = pd.DataFrame([
    {
        "Experiment": "Baseline",
        "Model": "Logistic Regression",
        "Accuracy": 0.7456,
        "Precision": 0.1379,
        "Sensitivity": 0.8000,
        "Specificity": 0.7428,
        "F1": 0.2353,
        "ROC_AUC": 0.8437,
        "PR_AUC": 0.2685,
        "MCC": 0.2597,
        "Brier": 0.1677,
        "ECE": 0.2694
    },
    {
        "Experiment": "Baseline",
        "Model": "Random Forest",
        "Accuracy": 0.9354,
        "Precision": 0.2143,
        "Sensitivity": 0.1200,
        "Specificity": 0.9774,
        "F1": 0.1538,
        "ROC_AUC": 0.7792,
        "PR_AUC": 0.1360,
        "MCC": 0.1287,
        "Brier": 0.0552,
        "ECE": 0.0455
    },
    {
        "Experiment": "Imbalance Handled",
        "Model": "XGBoost",
        "Accuracy": 0.8640,
        "Precision": 0.1799,
        "Sensitivity": 0.5000,
        "Specificity": 0.8827,
        "F1": 0.2646,
        "ROC_AUC": 0.8185,
        "PR_AUC": 0.2143,
        "MCC": 0.2408,
        "Brier": 0.0908,
        "ECE": 0.1270
    },
    {
        "Experiment": "Hyperparameter Tuned",
        "Model": "Tuned XGBoost",
        "Accuracy": 0.8885,
        "Precision": 0.2193,
        "Sensitivity": 0.5000,
        "Specificity": 0.9084,
        "F1": 0.3049,
        "ROC_AUC": 0.8129,
        "PR_AUC": 0.2176,
        "MCC": 0.2799,
        "Brier": 0.0817,
        "ECE": None
    },
    {
        "Experiment": "Stacking Ensemble",
        "Model": "Stacking Ensemble",
        "Accuracy": 0.9511,
        "Precision": 0.0000,
        "Sensitivity": 0.0000,
        "Specificity": 1.0000,
        "F1": 0.0000,
        "ROC_AUC": 0.8481,
        "PR_AUC": 0.2633,
        "MCC": 0.0000,
        "Brier": 0.0405,
        "ECE": None
    }
])

# ------------------------------------------------------------
# CALIBRATION RESULTS
# ------------------------------------------------------------

calibration_results = pd.DataFrame([
    ["Logistic Regression", "Base", 0.167651, 0.269417, 0.843663, 0.268538],
    ["Logistic Regression", "Sigmoid", 0.040808, 0.013076, 0.843498, 0.268718],
    ["Logistic Regression", "Isotonic", 0.040870, 0.005596, 0.839372, 0.279733],

    ["Random Forest", "Base", 0.055168, 0.045466, 0.779156, 0.136014],
    ["Random Forest", "Sigmoid", 0.044940, 0.009905, 0.797130, 0.146818],
    ["Random Forest", "Isotonic", 0.043900, 0.005279, 0.795494, 0.147444],

    ["XGBoost", "Base", 0.090797, 0.126977, 0.818477, 0.214350],
    ["XGBoost", "Sigmoid", 0.042305, 0.006552, 0.826132, 0.218543],
    ["XGBoost", "Isotonic", 0.041979, 0.008084, 0.823272, 0.236741]
], columns=[
    "Model",
    "Calibration_Method",
    "Brier_Score",
    "ECE",
    "ROC_AUC",
    "PR_AUC"
])

# ------------------------------------------------------------
# EXPLANATION STABILITY
# ------------------------------------------------------------

stability_results = pd.DataFrame([
    ["Age +1 year", 0.118741, 0.995511, 0.013589],
    ["Age -1 year", 0.112461, 0.992403, 0.020640],
    ["BMI +1", 0.174315, 0.986116, 0.027104],
    ["BMI -1", 0.235912, 0.938790, 0.050754],
    ["Glucose +5", 0.071934, 0.923061, 0.050003],
    ["Glucose -5", 0.138166, 0.995831, 0.018738]
], columns=[
    "Perturbation",
    "Probability",
    "Cosine_Similarity",
    "Mean_Absolute_SHAP_Difference"
])

# Add average stability
average_stability = pd.DataFrame([
    ["Average", None, 0.9720, 0.0301]
], columns=[
    "Perturbation",
    "Probability",
    "Cosine_Similarity",
    "Mean_Absolute_SHAP_Difference"
])

stability_results = pd.concat(
    [stability_results, average_stability],
    ignore_index=True
)

# ------------------------------------------------------------
# ENSEMBLE THRESHOLD RESULTS
# ------------------------------------------------------------

threshold_results = pd.DataFrame([
    [0.05, 0.755382, 0.145390, 0.82, 0.752058, 0.246988, 0.276069],
    [0.10, 0.855186, 0.211765, 0.72, 0.862140, 0.327273, 0.337211],
    [0.15, 0.906067, 0.270000, 0.54, 0.924897, 0.360000, 0.337528],
    [0.20, 0.934442, 0.345455, 0.38, 0.962963, 0.361905, 0.327847],
    [0.25, 0.948141, 0.440000, 0.22, 0.985597, 0.293333, 0.287089],
    [0.30, 0.948141, 0.285714, 0.04, 0.994856, 0.070175, 0.091162],
    [0.35, 0.951076, 0.000000, 0.00, 1.000000, 0.000000, 0.000000],
    [0.40, 0.951076, 0.000000, 0.00, 1.000000, 0.000000, 0.000000],
    [0.45, 0.951076, 0.000000, 0.00, 1.000000, 0.000000, 0.000000],
    [0.50, 0.951076, 0.000000, 0.00, 1.000000, 0.000000, 0.000000]
], columns=[
    "Threshold",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "MCC"
])

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

os.makedirs(results_dir, exist_ok=True)

model_results.to_csv(
    os.path.join(results_dir, "master_model_results.csv"),
    index=False
)

calibration_results.to_csv(
    os.path.join(results_dir, "master_calibration_results.csv"),
    index=False
)

stability_results.to_csv(
    os.path.join(results_dir, "master_stability_results.csv"),
    index=False
)

threshold_results.to_csv(
    os.path.join(results_dir, "master_threshold_results.csv"),
    index=False
)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("MASTER MODEL RESULTS")
print("=" * 80)

print(model_results.to_string(index=False))

print("\n" + "=" * 80)
print("CALIBRATION RESULTS")
print("=" * 80)

print(calibration_results.to_string(index=False))

print("\n" + "=" * 80)
print("EXPLANATION STABILITY")
print("=" * 80)

print(stability_results.to_string(index=False))

print("\n" + "=" * 80)
print("ENSEMBLE THRESHOLD ANALYSIS")
print("=" * 80)

print(threshold_results.to_string(index=False))

print("\n" + "=" * 80)
print("MASTER RESULTS CREATED SUCCESSFULLY")
print("=" * 80)

print("\nFiles created:")
print("results\\master_model_results.csv")
print("results\\master_calibration_results.csv")
print("results\\master_stability_results.csv")
print("results\\master_threshold_results.csv")