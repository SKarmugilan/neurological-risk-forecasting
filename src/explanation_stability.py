import os
import numpy as np
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
# 8. CONVERT TO DENSE ARRAYS
# ============================================================

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()


# ============================================================
# 9. FEATURE NAMES
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
# 11. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)

print(
    "SHAP explainer created successfully."
)


# ============================================================
# 12. SELECT PATIENT
# ============================================================

patient_index = 0

original_patient = X_test_processed[
    patient_index
].copy()

original_raw_patient = X_test.iloc[
    patient_index
]


# ============================================================
# 13. ORIGINAL PREDICTION
# ============================================================

original_probability = (
    model
    .predict_proba(
        original_patient.reshape(1, -1)
    )[0][1]
)


# ============================================================
# 14. ORIGINAL SHAP VALUES
# ============================================================

original_shap = explainer(
    original_patient.reshape(1, -1)
)

original_shap_values = (
    original_shap.values[0]
)


print(
    "\n" + "=" * 70
)

print(
    "ORIGINAL PATIENT"
)

print(
    "=" * 70
)

print(
    original_raw_patient.to_string()
)

print(
    "\nOriginal probability:",
    f"{original_probability:.4f}"
)


# ============================================================
# 15. CREATE PERTURBED PATIENTS
# ============================================================

perturbations = []

# Perturbation 1: age + 1 year
patient_1 = original_patient.copy()

age_position = list(feature_names).index(
    "num__age"
)

patient_1[age_position] += (
    1 / 10
)

perturbations.append(
    ("Age +1 year", patient_1)
)


# Perturbation 2: age - 1 year
patient_2 = original_patient.copy()

patient_2[age_position] -= (
    1 / 10
)

perturbations.append(
    ("Age -1 year", patient_2)
)


# Perturbation 3: BMI + 1
patient_3 = original_patient.copy()

bmi_position = list(feature_names).index(
    "num__bmi"
)

# Because BMI is standardized,
# use the training standard deviation
bmi_std = (
    preprocessor
    .named_transformers_["num"]
    .named_steps["scaler"]
    .scale_[
        numerical_features.index("bmi")
    ]
)

patient_3[bmi_position] += (
    1 / bmi_std
)

perturbations.append(
    ("BMI +1", patient_3)
)


# Perturbation 4: BMI -1
patient_4 = original_patient.copy()

patient_4[bmi_position] -= (
    1 / bmi_std
)

perturbations.append(
    ("BMI -1", patient_4)
)


# Perturbation 5: glucose +5
patient_5 = original_patient.copy()

glucose_position = list(
    feature_names
).index(
    "num__avg_glucose_level"
)

glucose_std = (
    preprocessor
    .named_transformers_["num"]
    .named_steps["scaler"]
    .scale_[
        numerical_features.index(
            "avg_glucose_level"
        )
    ]
)

patient_5[glucose_position] += (
    5 / glucose_std
)

perturbations.append(
    ("Glucose +5", patient_5)
)


# Perturbation 6: glucose -5
patient_6 = original_patient.copy()

patient_6[glucose_position] -= (
    5 / glucose_std
)

perturbations.append(
    ("Glucose -5", patient_6)
)


# ============================================================
# 16. CALCULATE SHAP FOR PERTURBED PATIENTS
# ============================================================

stability_results = []


for perturbation_name, patient in perturbations:

    probability = (
        model
        .predict_proba(
            patient.reshape(1, -1)
        )[0][1]
    )

    shap_result = explainer(
        patient.reshape(1, -1)
    )

    shap_values = (
        shap_result.values[0]
    )

    # Cosine similarity
    numerator = np.dot(
        original_shap_values,
        shap_values
    )

    denominator = (
        np.linalg.norm(
            original_shap_values
        )
        *
        np.linalg.norm(
            shap_values
        )
    )

    if denominator != 0:

        cosine_similarity = (
            numerator /
            denominator
        )

    else:

        cosine_similarity = 0


    # Mean absolute SHAP difference
    mean_absolute_difference = (
        np.mean(
            np.abs(
                original_shap_values -
                shap_values
            )
        )
    )


    stability_results.append({

        "Perturbation":
            perturbation_name,

        "Probability":
            probability,

        "Cosine_Similarity":
            cosine_similarity,

        "Mean_Absolute_SHAP_Difference":
            mean_absolute_difference

    })


# ============================================================
# 17. CREATE RESULTS DATAFRAME
# ============================================================

stability_df = pd.DataFrame(
    stability_results
)


# ============================================================
# 18. CALCULATE OVERALL STABILITY
# ============================================================

average_cosine_similarity = (
    stability_df[
        "Cosine_Similarity"
    ].mean()
)

average_shap_difference = (
    stability_df[
        "Mean_Absolute_SHAP_Difference"
    ].mean()
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPLANATION STABILITY RESULTS"
)

print(
    "=" * 70
)

print(
    stability_df.to_string(
        index=False
    )
)


print(
    "\nAverage SHAP cosine similarity:",
    f"{average_cosine_similarity:.4f}"
)

print(
    "Average mean absolute SHAP difference:",
    f"{average_shap_difference:.4f}"
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

stability_df.to_csv(
    "results/explanation_stability.csv",
    index=False
)


# ============================================================
# 21. CREATE STABILITY PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    stability_df["Perturbation"],
    stability_df["Cosine_Similarity"]
)

plt.axhline(
    average_cosine_similarity,
    linestyle="--",
    label="Average Similarity"
)

plt.xlabel(
    "Input Perturbation"
)

plt.ylabel(
    "SHAP Cosine Similarity"
)

plt.title(
    "SHAP Explanation Stability Under Small Perturbations"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.legend()

plt.tight_layout()


# ============================================================
# 22. SAVE PLOT
# ============================================================

plt.savefig(
    "results/explanation_stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPLANATION STABILITY ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFiles created:"
)

print(
    "results/explanation_stability.csv"
)

print(
    "results/explanation_stability.png"
)

print(
    "\nThe graph was saved automatically."
)