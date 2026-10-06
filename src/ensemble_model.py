import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "dataset/healthcare-dataset-stroke-data.csv"
)

print("Dataset shape:", data.shape)


# ============================================================
# 2. REMOVE ID
# ============================================================

data = data.drop(columns=["id"])


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["stroke"])
y = data["stroke"]


# ============================================================
# 4. FEATURE TYPES
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
# 5. PREPROCESSING
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
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts()
)


# ============================================================
# 7. DEFINE BASE MODELS
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
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


# ============================================================
# 8. CREATE INDIVIDUAL MODEL PIPELINES
# ============================================================

logistic_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        logistic_model
    )
])

rf_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        random_forest_model
    )
])

xgb_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        tuned_xgb_model
    )
])


# ============================================================
# 9. TRAIN INDIVIDUAL MODELS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING INDIVIDUAL MODELS"
)

print(
    "=" * 70
)


individual_models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": rf_pipeline,
    "Tuned XGBoost": xgb_pipeline
}


results = []


for model_name, model in individual_models.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model
        .predict_proba(X_test)[:, 1]
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred
        ).ravel()
    )

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    brier = brier_score_loss(
        y_test,
        y_probability
    )

    print(
        f"{model_name} completed."
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Sensitivity : {recall:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC      : {pr_auc:.4f}"
    )

    print(
        f"MCC         : {mcc:.4f}"
    )

    print(
        f"Brier Score : {brier:.4f}"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": recall,
        "Specificity": specificity,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "MCC": mcc,
        "Brier_Score": brier,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ============================================================
# 10. STACKING ENSEMBLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING STACKING ENSEMBLE"
)

print(
    "=" * 70
)


estimators = [

    (
        "logistic",
        Pipeline([
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ])
    ),

    (
        "random_forest",
        Pipeline([
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ])
    ),

    (
        "xgboost",
        Pipeline([
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                XGBClassifier(
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
            )
        ])
    )
]


ensemble = StackingClassifier(

    estimators=estimators,

    final_estimator=LogisticRegression(
        max_iter=1000
    ),

    cv=5,

    stack_method="predict_proba",

    passthrough=False,

    n_jobs=-1
)


# ============================================================
# 11. TRAIN ENSEMBLE
# ============================================================

ensemble.fit(
    X_train,
    y_train
)

print(
    "\nStacking ensemble trained successfully."
)


# ============================================================
# 12. ENSEMBLE PREDICTIONS
# ============================================================

ensemble_pred = ensemble.predict(
    X_test
)

ensemble_probability = (
    ensemble
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# 13. ENSEMBLE METRICS
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        ensemble_pred
    ).ravel()
)

accuracy = accuracy_score(
    y_test,
    ensemble_pred
)

precision = precision_score(
    y_test,
    ensemble_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    ensemble_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ensemble_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    ensemble_probability
)

pr_auc = average_precision_score(
    y_test,
    ensemble_probability
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

mcc = matthews_corrcoef(
    y_test,
    ensemble_pred
)

brier = brier_score_loss(
    y_test,
    ensemble_probability
)


# ============================================================
# 14. PRINT ENSEMBLE RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STACKING ENSEMBLE RESULTS"
)

print(
    "=" * 70
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Sensitivity : {recall:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"F1 Score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {roc_auc:.4f}"
)

print(
    f"PR-AUC      : {pr_auc:.4f}"
)

print(
    f"MCC         : {mcc:.4f}"
)

print(
    f"Brier Score : {brier:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        ensemble_pred
    )
)


# ============================================================
# 15. ADD ENSEMBLE TO RESULTS
# ============================================================

results.append({
    "Model": "Stacking Ensemble",
    "Accuracy": accuracy,
    "Precision": precision,
    "Sensitivity": recall,
    "Specificity": specificity,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "MCC": mcc,
    "Brier_Score": brier,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
})


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/ensemble_results.csv",
    index=False
)


# ============================================================
# 17. PRINT FINAL COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "INDIVIDUAL MODELS VS ENSEMBLE"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. COMPLETION MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ENSEMBLE EXPERIMENT COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFile created:"
)

print(
    "results/ensemble_results.csv"
)