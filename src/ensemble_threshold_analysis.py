import os
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
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
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


# ============================================================
# 7. DEFINE STACKING ENSEMBLE
# ============================================================

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
# 8. TRAIN ENSEMBLE
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

ensemble.fit(
    X_train,
    y_train
)

print(
    "\nStacking ensemble trained successfully."
)


# ============================================================
# 9. OBTAIN PROBABILITIES
# ============================================================

y_probability = (
    ensemble
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# 10. THRESHOLDS
# ============================================================

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


# ============================================================
# 11. STORE RESULTS
# ============================================================

results = []


# ============================================================
# 12. EVALUATE EACH THRESHOLD
# ============================================================

for threshold in thresholds:

    # Convert probability into class
    y_pred = (
        y_probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append({

        "Threshold":
            threshold,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Sensitivity":
            recall,

        "Specificity":
            specificity,

        "F1_Score":
            f1,

        "MCC":
            mcc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })


# ============================================================
# 13. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 14. GLOBAL DISCRIMINATION METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ENSEMBLE THRESHOLD ANALYSIS"
)

print(
    "=" * 70
)

print(
    f"\nROC-AUC : {roc_auc:.4f}"
)

print(
    f"PR-AUC  : {pr_auc:.4f}"
)


print(
    "\nThreshold-wise results:"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/ensemble_threshold_results.csv",
    index=False
)


# ============================================================
# 17. SAVE PROBABILITIES
# ============================================================

probability_df = pd.DataFrame({

    "Actual":
        y_test.to_numpy(),

    "Predicted_Probability":
        y_probability

})

probability_df.to_csv(
    "results/ensemble_test_probabilities.csv",
    index=False
)


# ============================================================
# 18. COMPLETION MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "THRESHOLD ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFiles created:"
)

print(
    "results/ensemble_threshold_results.csv"
)

print(
    "results/ensemble_test_probabilities.csv"
)