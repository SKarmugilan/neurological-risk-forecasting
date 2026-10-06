# Explainable AI (XAI) Package

> **Disclaimer**: This project is a research prototype and is not a clinically validated diagnostic system. It does not provide medical diagnosis or certified clinical evaluations.

---

## 1. Overview

The `explainability` package provides modular, reliable, and production-ready interpretability tools for early neurological risk forecasting. It translates complex non-linear machine learning predictions into human-interpretable feature attributions that reveal which physiological and demographic factors elevate or reduce predicted stroke risk for individual patients.

---

## 2. Core Methodological Distinctions

A foundational principle of this package is maintaining absolute mathematical and architectural rigor regarding **what model is being explained**:

| Property | TreeSHAP (`explain_patient_tree_shap`) | Model-Agnostic KernelSHAP (`explain_ensemble_kernel_shap`) | LIME (`explain_patient_lime`) |
| :--- | :--- | :--- | :--- |
| **Applicable Models** | Tree-based models only (Tuned XGBoost, Random Forest) | Any black-box model, specifically the **Stacking Ensemble** | Any model (Tuned XGBoost or Stacking Ensemble) |
| **Mathematical Basis** | Exact polynomial-time tree path conditional expectations | Game-theoretic Shapley values approximated via weighted linear regression | Local linear surrogate model fitted on perturbed instances |
| **Computational Speed** | Extremely fast ($< 50$ ms) | Moderate ($0.2$–$0.5$ s with $N=25$ background references) | Fast ($0.1$–$0.2$ s with 1,000 samples) |
| **Theoretical Guarantees** | Efficiency, Symmetry, Linearity, Additivity | Efficiency, Symmetry, Linearity, Additivity (asymptotically) | Local fidelity without global additivity |
| **Ensemble Compatibility** | **Incompatible** with heterogeneous stacking ensembles | **Fully compatible** with stacking ensemble | **Fully compatible** with stacking ensemble |

### Explaining XGBoost vs. Explaining the Final Stacking Ensemble

* **TreeSHAP for XGBoost**: Evaluates split gains and path leaves strictly within XGBoost's 200 boosted trees. It explains the single best gradient boosted estimator.
* **Stacking Ensemble Explanation**: The StackingClassifier combines Logistic Regression, Random Forest, and Tuned XGBoost through a Logistic Regression meta-learner. Because the ensemble is heterogeneous and contains non-tree estimators, **TreeSHAP cannot be applied to the ensemble**. Instead, `explain_ensemble_kernel_shap` evaluates the complete ensemble probability output over background reference samples.
* **Integrity Guarantee**: This package strictly distinguishes between individual model explanations and ensemble explanations, never substituting one for the other.

---

## 3. Package Structure

```
explainability/
├── __init__.py           # Unified public API exports
├── utils.py              # Input validation, data conversion, and formatting helpers
├── shap_explainer.py     # TreeSHAP (XGBoost/RF) and Model-Agnostic KernelSHAP (Ensemble)
├── lime_explainer.py     # Local Interpretable Model-agnostic Explanations (LIME)
├── stability.py          # Perturbation robustness & cosine similarity analysis
└── README.md             # Technical documentation and API integration guide
```

---

## 4. Key Functions and Usage

### 1. Patient Input Format
All explainer functions accept standard Python dictionaries containing raw patient screening features:

```python
patient_record = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}
```

---

### 2. Local TreeSHAP Explanation (Tuned XGBoost)

```python
from explainability import explain_patient_tree_shap

result = explain_patient_tree_shap(
    patient_data=patient_record,
    model_name="tuned_xgboost",
    save_plot_path="results/sample_patient_shap.png"  # Optional
)

print(f"Predicted Probability: {result['predicted_stroke_probability']:.4f}")
for item in result["feature_contributions"][:5]:
    print(f"  {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
```

---

### 3. Model-Agnostic KernelSHAP (Stacking Ensemble)

```python
from explainability import explain_ensemble_kernel_shap

ensemble_result = explain_ensemble_kernel_shap(
    patient_data=patient_record,
    background_samples=25,  # Lightweight for low-latency web responses
    nsamples=60,
    save_plot_path="results/sample_ensemble_shap.png"
)

print(f"Ensemble Risk Probability: {ensemble_result['predicted_stroke_probability']:.4f}")
for item in ensemble_result["feature_contributions"][:5]:
    print(f"  {item['feature']}: {item['contribution']:+.4f}")
```

---

### 4. LIME Local Explanation

```python
from explainability import explain_patient_lime

lime_result = explain_patient_lime(
    patient_data=patient_record,
    model_name="tuned_xgboost", # or 'stacking_ensemble'
    num_features=10,
    num_samples=1000,
    random_state=42
)

for item in lime_result["top_feature_contributions"][:5]:
    print(f"  {item['feature']}: {item['contribution']:+.4f} ({item['direction']})")
```

---

### 5. Explanation Stability Analysis

Evaluates explanation consistency under 6 standard physiological perturbations (Age $\pm 1$, BMI $\pm 1$, Glucose $\pm 5$):

```python
from explainability import evaluate_explanation_stability

stability_result = evaluate_explanation_stability(
    patient_data=patient_record,
    model_name="tuned_xgboost"
)

print(f"Average Cosine Similarity: {stability_result['average_cosine_similarity']:.4f}")
print(f"Mean Absolute SHAP Shift: {stability_result['average_mean_absolute_shap_difference']:.6f}")
```

---

## 5. Integration with Future FastAPI Backend

The package is formatted for direct integration into REST API route handlers:

```python
from fastapi import FastAPI, HTTPException
from explainability import (
    validate_patient_dict,
    explain_patient_tree_shap,
    explain_ensemble_kernel_shap
)

app = FastAPI()

@app.post("/api/explain/xgboost")
async def explain_xgboost_endpoint(patient_payload: dict):
    try:
        clean_patient = validate_patient_dict(patient_payload)
        explanation = explain_patient_tree_shap(clean_patient)
        return {"status": "success", "data": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/explain/ensemble")
async def explain_ensemble_endpoint(patient_payload: dict):
    try:
        clean_patient = validate_patient_dict(patient_payload)
        explanation = explain_ensemble_kernel_shap(clean_patient, background_samples=25, nsamples=60)
        return {"status": "success", "data": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## 6. Technical Limitations

1. **Correlation Collinearity**: When strong collinearity exists among input features (e.g. Age and Hypertension), additive attributions distribute importance across correlated features rather than isolating causality.
2. **Local Surrogate Sampling**: LIME relies on local Gaussian perturbation sampling around the query point; local boundaries can vary slightly between random seeds if sample sizes are insufficient.
3. **KernelSHAP Computational Cost**: Model-agnostic KernelSHAP scales with the number of background samples and perturbation evaluations. While lightweight configurations ($N=25$) run in $<0.3$s, exhaustive sampling requires substantially higher compute.
4. **Non-Causal Interpretability**: Explanations reflect model-learned correlations within retrospective data, not medical etiology or clinical causation.
