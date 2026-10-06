import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


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
# 4. DEFINE FEATURES
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
# 10. CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 11. STORAGE FOR CALIBRATION RESULTS
# ============================================================

calibration_results = []


# ============================================================
# 12. TRAIN MODELS AND CALCULATE CALIBRATION
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predicted probabilities
    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Brier Score
    brier_score = brier_score_loss(
        y_test,
        y_probability
    )

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
        y_probability,
        n_bins=10,
        strategy="uniform"
    )

    print(f"Brier Score : {brier_score:.4f}")

    print("\nCalibration points:")

    for predicted, actual in zip(
        mean_predicted_value,
        fraction_of_positives
    ):

        print(
            f"Predicted probability: {predicted:.4f} "
            f"| Actual frequency: {actual:.4f}"
        )

        calibration_results.append({
            "Model": model_name,
            "Mean_Predicted_Probability": predicted,
            "Fraction_of_Positive_Cases": actual,
            "Brier_Score": brier_score
        })


# ============================================================
# 13. SAVE CALIBRATION DATA
# ============================================================

calibration_df = pd.DataFrame(
    calibration_results
)

calibration_df.to_csv(
    "results/calibration_results.csv",
    index=False
)


print("\n" + "=" * 60)
print("CALIBRATION DATA SAVED")
print("=" * 60)

print(
    "\nCSV file: results/calibration_results.csv"
)


# ============================================================
# 14. CREATE CALIBRATION PLOT
# ============================================================

plt.figure(figsize=(9, 7))


# Perfect calibration line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)


# Plot each model
for model_name in models.keys():

    model_data = calibration_df[
        calibration_df["Model"] == model_name
    ]

    plt.plot(
        model_data["Mean_Predicted_Probability"],
        model_data["Fraction_of_Positive_Cases"],
        marker="o",
        label=model_name
    )


# ============================================================
# 15. FORMAT GRAPH
# ============================================================

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Fraction of Positive Cases"
)

plt.title(
    "Calibration Curves for Stroke Risk Prediction Models"
)

plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


# ============================================================
# 16. SAVE GRAPH
# ============================================================

plt.savefig(
    "results/calibration_curves.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print("CALIBRATION CURVE SAVED")

print("=" * 60)

print(
    "\nImage file:"
)

print(
    "results/calibration_curves.png"
)