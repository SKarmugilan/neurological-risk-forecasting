import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss
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
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["stroke"])
y = data["stroke"]


# ============================================================
# 4. DEFINE FEATURE GROUPS
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
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ============================================================
# 7. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
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


print("\nTraining class distribution:")
print(y_train.value_counts())


print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 9. DEFINE MODELS
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
# 10. EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(y_true, y_probability, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Lower ECE indicates better probability calibration.
    """

    bin_edges = [i / n_bins for i in range(n_bins + 1)]

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

        # Skip empty bins
        if mask.sum() == 0:
            continue

        # Actual fraction of positive cases
        bin_accuracy = y_true[mask].mean()

        # Average predicted probability
        bin_confidence = y_probability[mask].mean()

        # Proportion of samples in this bin
        bin_weight = mask.sum() / len(y_true)

        # Add weighted calibration error
        ece += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return ece


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)


    # --------------------------------------------------------
    # Create complete preprocessing + model pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )


    # Probability of stroke class
    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    # ========================================================
    # 12. CLASSIFICATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    pr_auc = average_precision_score(
        y_test,
        y_probability
    )


    # ========================================================
    # 13. MEDICAL-RISK METRICS
    # ========================================================

    # Sensitivity
    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    # ========================================================
    # 14. MATTHEWS CORRELATION COEFFICIENT
    # ========================================================

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    # ========================================================
    # 15. CALIBRATION METRICS
    # ========================================================

    # Brier Score
    brier_score = brier_score_loss(
        y_test,
        y_probability
    )


    # Expected Calibration Error
    ece = calculate_ece(
        y_test.to_numpy(),
        y_probability
    )


    # ========================================================
    # 16. PRINT RESULTS
    # ========================================================

    print(f"Accuracy    : {accuracy:.4f}")

    print(f"Precision   : {precision:.4f}")

    print(f"Recall      : {recall:.4f}")

    print(f"Sensitivity : {sensitivity:.4f}")

    print(f"Specificity : {specificity:.4f}")

    print(f"F1 Score    : {f1:.4f}")

    print(f"ROC-AUC     : {roc_auc:.4f}")

    print(f"PR-AUC      : {pr_auc:.4f}")

    print(f"MCC         : {mcc:.4f}")

    print(f"Brier Score : {brier_score:.4f}")

    print(f"ECE         : {ece:.4f}")


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    # ========================================================
    # 17. STORE RESULTS
    # ========================================================

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "Sensitivity": sensitivity,

        "Specificity": specificity,

        "F1_Score": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "MCC": mcc,

        "Brier_Score": brier_score,

        "ECE": ece,

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    })


# ============================================================
# 18. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 19. CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 20. SAVE RESULTS TO CSV
# ============================================================

results_df.to_csv(
    "results/model_results.csv",
    index=False
)


# ============================================================
# 21. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)

print("RESULTS SAVED")

print("=" * 60)


print("\nCSV file:")

print(
    "results/model_results.csv"
)


print("\nFinal Results Table:")

print(
    results_df.to_string(
        index=False
    )
)