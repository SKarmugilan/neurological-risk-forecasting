import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# ACTUAL ROC AND PRECISION-RECALL CURVES
# ============================================================

RESULTS_DIR = "results"

input_file = os.path.join(
    RESULTS_DIR,
    "ensemble_test_probabilities.csv"
)

# ------------------------------------------------------------
# LOAD ACTUAL TEST PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("=" * 70)
print("ACTUAL ENSEMBLE ROC / PRECISION-RECALL ANALYSIS")
print("=" * 70)

print("\nColumns found:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

# ------------------------------------------------------------
# IDENTIFY TARGET AND PROBABILITY COLUMNS
# ------------------------------------------------------------

# Expected columns from E12:
# Actual
# Probability

target_column = None
probability_column = None

for column in df.columns:

    column_lower = column.lower()

    if column_lower in [
        "actual",
        "actual_label",
        "true_label",
        "y_true",
        "stroke"
    ]:
        target_column = column

    if column_lower in [
        "probability",
        "predicted_probability",
        "prob",
        "y_prob"
    ]:
        probability_column = column

if target_column is None:
    raise ValueError(
        "Could not identify the actual target column."
    )

if probability_column is None:
    raise ValueError(
        "Could not identify the probability column."
    )

print("\nTarget column:", target_column)
print("Probability column:", probability_column)

y_true = df[target_column]
y_prob = df[probability_column]

# ------------------------------------------------------------
# ACTUAL METRICS
# ------------------------------------------------------------

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
)

print("\nActual ROC-AUC:", round(roc_auc, 4))
print("Actual PR-AUC :", round(pr_auc, 4))

# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Stacking Ensemble (ROC-AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

roc_file = os.path.join(
    RESULTS_DIR,
    "actual_roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# PRECISION-RECALL CURVE
# ------------------------------------------------------------

precision, recall, pr_thresholds = precision_recall_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    recall,
    precision,
    label=f"Stacking Ensemble (PR-AUC = {pr_auc:.4f})"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

pr_file = os.path.join(
    RESULTS_DIR,
    "actual_precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# SAVE ROC DATA
# ------------------------------------------------------------

roc_df = pd.DataFrame({
    "False_Positive_Rate": fpr,
    "True_Positive_Rate": tpr,
    "Threshold": roc_thresholds
})

roc_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "actual_roc_curve_data.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# SAVE PR DATA
# ------------------------------------------------------------

pr_df = pd.DataFrame({
    "Precision": precision,
    "Recall": recall
})

pr_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "actual_precision_recall_curve_data.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ACTUAL ROC / PR CURVES GENERATED")
print("=" * 70)

print("\nFiles created:")

print(
    "results\\actual_roc_curve.png"
)

print(
    "results\\actual_precision_recall_curve.png"
)

print(
    "results\\actual_roc_curve_data.csv"
)

print(
    "results\\actual_precision_recall_curve_data.csv"
)

print("\n" + "=" * 70)