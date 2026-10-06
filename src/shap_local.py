import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. PREPROCESS DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 8. FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "\nNumber of processed features:",
    len(feature_names)
)


# ============================================================
# 9. TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19.54,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)

print(
    "\nXGBoost model trained successfully."
)


# ============================================================
# 10. SELECT ONE TEST PATIENT
# ============================================================

patient_index = 0

patient_processed = (
    X_test_processed[patient_index]
)

patient_original = X_test.iloc[
    patient_index
]

actual_label = y_test.iloc[
    patient_index
]


# ============================================================
# 11. PATIENT PREDICTION
# ============================================================

patient_probability = (
    model
    .predict_proba(
        patient_processed.reshape(1, -1)
    )[0][1]
)

patient_prediction = int(
    patient_probability >= 0.5
)


print("\n" + "=" * 70)
print("INDIVIDUAL PATIENT ANALYSIS")
print("=" * 70)

print(
    "\nSelected test patient index:",
    patient_index
)

print(
    "\nPatient feature values:"
)

print(
    patient_original.to_string()
)

print(
    "\nActual stroke label:",
    actual_label
)

print(
    "Predicted stroke-risk probability:",
    f"{patient_probability:.4f}"
)

print(
    "Predicted class:",
    patient_prediction
)


# ============================================================
# 12. SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer(
    patient_processed.reshape(1, -1)
)


print(
    "\nSHAP explanation calculated successfully."
)


# ============================================================
# 13. EXTRACT SHAP VALUES
# ============================================================

patient_shap_values = (
    shap_values.values[0]
)


# ============================================================
# 14. CREATE LOCAL SHAP DATAFRAME
# ============================================================

local_importance = pd.DataFrame({

    "Feature": feature_names,

    "SHAP_Value": patient_shap_values,

    "Absolute_SHAP": abs(
        patient_shap_values
    )

})


# ============================================================
# 15. SORT BY ABSOLUTE SHAP
# ============================================================

local_importance = (
    local_importance
    .sort_values(
        by="Absolute_SHAP",
        ascending=False
    )
)


# ============================================================
# 16. SAVE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 17. SAVE LOCAL SHAP CSV
# ============================================================

local_importance.to_csv(
    "results/shap_local_patient.csv",
    index=False
)


# ============================================================
# 18. PRINT TOP LOCAL FEATURES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP FEATURES FOR THIS PATIENT"
)

print(
    "=" * 70
)

print(
    local_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 19. CREATE LOCAL BAR PLOT
# ============================================================

top_features = (
    local_importance
    .head(10)
    .sort_values(
        by="SHAP_Value"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["SHAP_Value"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Local SHAP Explanation - Individual Patient"
)

plt.tight_layout()


# ============================================================
# 20. SAVE LOCAL SHAP PLOT
# ============================================================

plt.savefig(
    "results/shap_local_patient.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. SHAP WATERFALL DATA
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SHAP DIRECTION INTERPRETATION"
)

print(
    "=" * 70
)

positive_features = (
    local_importance[
        local_importance["SHAP_Value"] > 0
    ]
)

negative_features = (
    local_importance[
        local_importance["SHAP_Value"] < 0
    ]
)


print(
    "\nFeatures pushing prediction higher:"
)

if len(positive_features) > 0:

    print(
        positive_features
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "None"
    )


print(
    "\nFeatures pushing prediction lower:"
)

if len(negative_features) > 0:

    print(
        negative_features
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "None"
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOCAL SHAP ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFiles created:"
)

print(
    "results/shap_local_patient.csv"
)

print(
    "results/shap_local_patient.png"
)

print(
    "\nThe graph was saved automatically."
)