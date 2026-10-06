import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from lime.lime_tabular import LimeTabularExplainer


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
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


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
# 8. CONVERT TO DENSE ARRAY
# ============================================================

X_train_processed = (
    X_train_processed.toarray()
    if hasattr(
        X_train_processed,
        "toarray"
    )
    else X_train_processed
)

X_test_processed = (
    X_test_processed.toarray()
    if hasattr(
        X_test_processed,
        "toarray"
    )
    else X_test_processed
)


# ============================================================
# 9. GET FEATURE NAMES
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
# 10. TRAIN XGBOOST
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
# 11. SELECT SAME PATIENT AS E8
# ============================================================

patient_index = 0

patient = X_test_processed[
    patient_index
]

patient_original = X_test.iloc[
    patient_index
]

actual_label = y_test.iloc[
    patient_index
]


# ============================================================
# 12. PREDICTION
# ============================================================

patient_probability = (
    model
    .predict_proba(
        patient.reshape(1, -1)
    )[0][1]
)

patient_prediction = int(
    patient_probability >= 0.5
)


print(
    "\n" + "=" * 70
)

print(
    "LIME INDIVIDUAL PATIENT ANALYSIS"
)

print(
    "=" * 70
)

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
# 13. CREATE LIME EXPLAINER
# ============================================================

explainer = LimeTabularExplainer(
    training_data=X_train_processed,
    feature_names=feature_names,
    class_names=[
        "No Stroke",
        "Stroke"
    ],
    mode="classification",
    discretize_continuous=False,
    random_state=42
)


print(
    "\nLIME explainer created successfully."
)


# ============================================================
# 14. GENERATE LIME EXPLANATION
# ============================================================

explanation = explainer.explain_instance(
    patient,
    model.predict_proba,
    num_features=15,
    num_samples=5000
)


print(
    "LIME explanation calculated successfully."
)


# ============================================================
# 15. GET EXPLANATION RESULTS
# ============================================================

lime_results = explanation.as_list(
    label=1
)


# ============================================================
# 16. CREATE DATAFRAME
# ============================================================

lime_df = pd.DataFrame(
    lime_results,
    columns=[
        "Feature",
        "LIME_Weight"
    ]
)

lime_df["Absolute_LIME"] = (
    lime_df["LIME_Weight"].abs()
)

lime_df = lime_df.sort_values(
    by="Absolute_LIME",
    ascending=False
)


# ============================================================
# 17. CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 18. SAVE LIME RESULTS
# ============================================================

lime_df.to_csv(
    "results/lime_local_patient.csv",
    index=False
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP LIME FEATURES FOR THIS PATIENT"
)

print(
    "=" * 70
)

print(
    lime_df.to_string(
        index=False
    )
)


# ============================================================
# 20. CREATE LIME BAR PLOT
# ============================================================

plot_df = (
    lime_df
    .head(10)
    .sort_values(
        by="LIME_Weight"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_df["Feature"],
    plot_df["LIME_Weight"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "LIME Weight"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "LIME Local Explanation - Individual Patient"
)

plt.tight_layout()


# ============================================================
# 21. SAVE LIME PLOT
# ============================================================

plt.savefig(
    "results/lime_local_patient.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. POSITIVE CONTRIBUTIONS
# ============================================================

positive_lime = lime_df[
    lime_df["LIME_Weight"] > 0
]


# ============================================================
# 23. NEGATIVE CONTRIBUTIONS
# ============================================================

negative_lime = lime_df[
    lime_df["LIME_Weight"] < 0
]


# ============================================================
# 24. DISPLAY DIRECTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LIME DIRECTION INTERPRETATION"
)

print(
    "=" * 70
)


print(
    "\nFeatures pushing prediction toward STROKE:"
)

if len(positive_lime) > 0:

    print(
        positive_lime
        .to_string(index=False)
    )

else:

    print("None")


print(
    "\nFeatures pushing prediction toward NO STROKE:"
)

if len(negative_lime) > 0:

    print(
        negative_lime
        .to_string(index=False)
    )

else:

    print("None")


# ============================================================
# 25. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LIME LOCAL ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFiles created:"
)

print(
    "results/lime_local_patient.csv"
)

print(
    "results/lime_local_patient.png"
)

print(
    "\nThe graph was saved automatically."
)