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
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
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

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 8. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print("\nNumber of processed features:", len(feature_names))


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

print("\nXGBoost model trained successfully.")


# ============================================================
# 10. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_test_processed
)

print("SHAP values calculated successfully.")


# ============================================================
# 11. CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 12. SHAP SUMMARY BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title(
    "Global SHAP Feature Importance - XGBoost"
)

plt.tight_layout()

plt.savefig(
    "results/shap_global_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. SHAP SUMMARY DOT PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Summary Plot - XGBoost"
)

plt.tight_layout()

plt.savefig(
    "results/shap_global_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. CALCULATE MEAN ABSOLUTE SHAP VALUES
# ============================================================

mean_abs_shap = abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)


# ============================================================
# 15. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "results/shap_global_importance.csv",
    index=False
)


# ============================================================
# 16. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SHAP GLOBAL ANALYSIS COMPLETED")
print("=" * 60)

print("\nFiles created:")

print(
    "results/shap_global_bar.png"
)

print(
    "results/shap_global_summary.png"
)

print(
    "results/shap_global_importance.csv"
)