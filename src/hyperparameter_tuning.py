import os
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
# 7. BASE XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    scale_pos_weight=19.54,
    eval_metric="logloss",
    random_state=42
)


# ============================================================
# 8. PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        xgb_model
    )
])


# ============================================================
# 9. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_grid = {

    "model__n_estimators": [
        100,
        200,
        300,
        500
    ],

    "model__max_depth": [
        2,
        3,
        4,
        5,
        6
    ],

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__min_child_weight": [
        1,
        3,
        5
    ]
}


# ============================================================
# 10. RANDOMIZED SEARCH
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING HYPERPARAMETER OPTIMIZATION"
)

print(
    "=" * 70
)

print(
    "\nThis may take some time..."
)


search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=20,

    scoring="average_precision",

    cv=5,

    verbose=1,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 11. RUN SEARCH
# ============================================================

search.fit(
    X_train,
    y_train
)


# ============================================================
# 12. BEST PARAMETERS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST HYPERPARAMETERS"
)

print(
    "=" * 70
)

print(
    search.best_params_
)


print(
    "\nBest Cross-Validation PR-AUC:"
)

print(
    f"{search.best_score_:.4f}"
)


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 14. TEST SET PREDICTIONS
# ============================================================

y_pred = best_model.predict(
    X_test
)

y_probability = (
    best_model
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        y_pred
    ).ravel()
)


# ============================================================
# 16. METRICS
# ============================================================

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


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TUNED XGBOOST TEST RESULTS"
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
    f"Recall      : {recall:.4f}"
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
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 18. SAVE BEST PARAMETERS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

best_parameters = pd.DataFrame(
    [
        search.best_params_
    ]
)

best_parameters.to_csv(
    "results/best_xgboost_parameters.csv",
    index=False
)


# ============================================================
# 19. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Sensitivity",
        "Specificity",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC",
        "MCC",
        "TN",
        "FP",
        "FN",
        "TP"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        recall,
        specificity,
        f1,
        roc_auc,
        pr_auc,
        mcc,
        tn,
        fp,
        fn,
        tp
    ]
})

test_results.to_csv(
    "results/tuned_xgboost_results.csv",
    index=False
)


# ============================================================
# 20. SAVE ALL SEARCH RESULTS
# ============================================================

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results.to_csv(
    "results/hyperparameter_search_results.csv",
    index=False
)


# ============================================================
# 21. COMPLETION MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "HYPERPARAMETER OPTIMIZATION COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFiles created:"
)

print(
    "results/best_xgboost_parameters.csv"
)

print(
    "results/tuned_xgboost_results.csv"
)

print(
    "results/hyperparameter_search_results.csv"
)