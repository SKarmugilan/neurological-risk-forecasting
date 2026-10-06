import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "dataset/healthcare-dataset-stroke-data.csv"
)

print("Dataset shape:", data.shape)


# ============================================================
# 2. REMOVE ID COLUMN
# ============================================================

data = data.drop(columns=["id"])


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["stroke"])
y = data["stroke"]


# ============================================================
# 4. FEATURE GROUPS
# ============================================================

numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]


# ============================================================
# 5. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ============================================================
# 7. COMPLETE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([
    (
        "num",
        numerical_pipeline,
        numerical_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 9. DEFINE BASE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=19.54,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 10. ECE FUNCTION
# ============================================================

def calculate_ece(
    y_true,
    y_probability,
    n_bins=10
):

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (y_probability >= lower) &
                (y_probability <= upper)
            )

        else:

            mask = (
                (y_probability >= lower) &
                (y_probability < upper)
            )

        if mask.sum() == 0:
            continue

        bin_accuracy = y_true[mask].mean()

        bin_confidence = y_probability[mask].mean()

        bin_weight = mask.sum() / len(y_true)

        ece += (
            bin_weight *
            abs(
                bin_accuracy -
                bin_confidence
            )
        )

    return ece


# ============================================================
# 11. STORE RESULTS
# ============================================================

results = []


# ============================================================
# 12. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 13. CALIBRATION CURVE DATA
# ============================================================

calibration_data = []


# ============================================================
# 14. TRAIN AND CALIBRATE EACH MODEL
# ============================================================

for model_name, base_model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)


    # --------------------------------------------------------
    # BASE PIPELINE
    # --------------------------------------------------------

    base_pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            base_model
        )
    ])


    # --------------------------------------------------------
    # FIT BASE MODEL
    # --------------------------------------------------------

    base_pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # BASE MODEL PREDICTIONS
    # --------------------------------------------------------

    base_probability = (
        base_pipeline
        .predict_proba(X_test)[:, 1]
    )


    # --------------------------------------------------------
    # BASE MODEL METRICS
    # --------------------------------------------------------

    base_brier = brier_score_loss(
        y_test,
        base_probability
    )

    base_roc_auc = roc_auc_score(
        y_test,
        base_probability
    )

    base_pr_auc = average_precision_score(
        y_test,
        base_probability
    )

    base_ece = calculate_ece(
        y_test,
        base_probability
    )


    print("\nBASE MODEL")
    print("-" * 50)

    print(
        f"Brier Score : {base_brier:.4f}"
    )

    print(
        f"ROC-AUC     : {base_roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {base_pr_auc:.4f}"
    )

    print(
        f"ECE         : {base_ece:.4f}"
    )


    # ========================================================
    # 15. SIGMOID CALIBRATION
    # ========================================================

    sigmoid_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5
    )

    sigmoid_model.fit(
        X_train,
        y_train
    )

    sigmoid_probability = (
        sigmoid_model
        .predict_proba(X_test)[:, 1]
    )


    sigmoid_brier = brier_score_loss(
        y_test,
        sigmoid_probability
    )

    sigmoid_roc_auc = roc_auc_score(
        y_test,
        sigmoid_probability
    )

    sigmoid_pr_auc = average_precision_score(
        y_test,
        sigmoid_probability
    )

    sigmoid_ece = calculate_ece(
        y_test,
        sigmoid_probability
    )


    print("\nSIGMOID CALIBRATION")
    print("-" * 50)

    print(
        f"Brier Score : {sigmoid_brier:.4f}"
    )

    print(
        f"ROC-AUC     : {sigmoid_roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {sigmoid_pr_auc:.4f}"
    )

    print(
        f"ECE         : {sigmoid_ece:.4f}"
    )


    # ========================================================
    # 16. ISOTONIC CALIBRATION
    # ========================================================

    isotonic_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="isotonic",
        cv=5
    )

    isotonic_model.fit(
        X_train,
        y_train
    )

    isotonic_probability = (
        isotonic_model
        .predict_proba(X_test)[:, 1]
    )


    isotonic_brier = brier_score_loss(
        y_test,
        isotonic_probability
    )

    isotonic_roc_auc = roc_auc_score(
        y_test,
        isotonic_probability
    )

    isotonic_pr_auc = average_precision_score(
        y_test,
        isotonic_probability
    )

    isotonic_ece = calculate_ece(
        y_test,
        isotonic_probability
    )


    print("\nISOTONIC CALIBRATION")
    print("-" * 50)

    print(
        f"Brier Score : {isotonic_brier:.4f}"
    )

    print(
        f"ROC-AUC     : {isotonic_roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {isotonic_pr_auc:.4f}"
    )

    print(
        f"ECE         : {isotonic_ece:.4f}"
    )


    # ========================================================
    # 17. SAVE RESULTS
    # ========================================================

    results.append({

        "Model": model_name,

        "Calibration_Method": "Base",

        "Brier_Score": base_brier,

        "ECE": base_ece,

        "ROC_AUC": base_roc_auc,

        "PR_AUC": base_pr_auc

    })


    results.append({

        "Model": model_name,

        "Calibration_Method": "Sigmoid",

        "Brier_Score": sigmoid_brier,

        "ECE": sigmoid_ece,

        "ROC_AUC": sigmoid_roc_auc,

        "PR_AUC": sigmoid_pr_auc

    })


    results.append({

        "Model": model_name,

        "Calibration_Method": "Isotonic",

        "Brier_Score": isotonic_brier,

        "ECE": isotonic_ece,

        "ROC_AUC": isotonic_roc_auc,

        "PR_AUC": isotonic_pr_auc

    })


    # ========================================================
    # 18. CALIBRATION CURVES
    # ========================================================

    base_fraction, base_mean = calibration_curve(
        y_test,
        base_probability,
        n_bins=10,
        strategy="uniform"
    )

    sigmoid_fraction, sigmoid_mean = calibration_curve(
        y_test,
        sigmoid_probability,
        n_bins=10,
        strategy="uniform"
    )

    isotonic_fraction, isotonic_mean = calibration_curve(
        y_test,
        isotonic_probability,
        n_bins=10,
        strategy="uniform"
    )


    calibration_data.append({

        "Model": model_name,

        "Base": (
            base_mean,
            base_fraction
        ),

        "Sigmoid": (
            sigmoid_mean,
            sigmoid_fraction
        ),

        "Isotonic": (
            isotonic_mean,
            isotonic_fraction
        )

    })


# ============================================================
# 19. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 20. SAVE RESULTS CSV
# ============================================================

results_df.to_csv(
    "results/calibrated_model_results.csv",
    index=False
)


# ============================================================
# 21. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED MODEL RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 22. CREATE CALIBRATION PLOT
# ============================================================

plt.figure(
    figsize=(10, 8)
)


for item in calibration_data:

    model_name = item["Model"]

    base_mean, base_fraction = item["Base"]

    sigmoid_mean, sigmoid_fraction = item["Sigmoid"]

    isotonic_mean, isotonic_fraction = item["Isotonic"]


    plt.plot(
        base_mean,
        base_fraction,
        marker="o",
        linestyle="--",
        label=model_name + " - Base"
    )

    plt.plot(
        sigmoid_mean,
        sigmoid_fraction,
        marker="s",
        linestyle="-",
        label=model_name + " - Sigmoid"
    )

    plt.plot(
        isotonic_mean,
        isotonic_fraction,
        marker="^",
        linestyle=":",
        label=model_name + " - Isotonic"
    )


# Perfect calibration line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)


plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Fraction of Positives"
)

plt.title(
    "Calibration Curves - Before and After Calibration"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 23. SAVE FIGURE
# ============================================================

plt.savefig(
    "results/calibrated_calibration_curves.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 24. CLOSE FIGURE
# ============================================================

plt.close()


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    "results/calibrated_model_results.csv"
)

print(
    "results/calibrated_calibration_curves.png"
)

print(
    "\nThe graph was saved automatically."
)

print(
    "No plt.show() is used, so the program will exit normally."
)