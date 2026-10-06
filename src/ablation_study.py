import os
import pandas as pd

from sklearn.model_selection import train_test_split
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
    brier_score_loss,
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
# 7. ECE FUNCTION
# ============================================================

def calculate_ece(
    y_true,
    y_probability,
    n_bins=10
):

    y_true = y_true.to_numpy()

    ece = 0.0

    bin_edges = [
        i / n_bins
        for i in range(n_bins + 1)
    ]

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

        bin_accuracy = (
            y_true[mask].mean()
        )

        bin_confidence = (
            y_probability[mask].mean()
        )

        bin_weight = (
            mask.sum() /
            len(y_true)
        )

        ece += (
            bin_weight *
            abs(
                bin_accuracy -
                bin_confidence
            )
        )

    return ece


# ============================================================
# 8. DEFINE ABLATION MODELS
# ============================================================

models = {

    "XGBoost - No Imbalance Handling":

        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),


    "XGBoost - Imbalance Handling":

        XGBClassifier(
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
# 9. RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# 10. TRAIN EACH ABLATION MODEL
# ============================================================

for model_name, model in models.items():

    print(
        "\n" + "=" * 70
    )

    print(
        model_name
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred
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

    ece = calculate_ece(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

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
        f"Brier Score : {brier:.4f}"
    )

    print(
        f"ECE         : {ece:.4f}"
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


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({

        "Experiment":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "Sensitivity":
            recall,

        "Specificity":
            specificity,

        "F1_Score":
            f1,

        "ROC_AUC":
            roc_auc,

        "PR_AUC":
            pr_auc,

        "MCC":
            mcc,

        "Brier_Score":
            brier,

        "ECE":
            ece,

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
# 11. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/ablation_results.csv",
    index=False
)


# ============================================================
# 13. PRINT FINAL TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ABLATION STUDY RESULTS"
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
# 14. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ABLATION STUDY COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFile created:"
)

print(
    "results/ablation_results.csv"
)