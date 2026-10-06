"""
explainability/stability.py - Explanation Stability Evaluation Module
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

Measures the mathematical robustness and consistency of local explanations under
controlled physiological perturbations (Age ±1, BMI ±1, Glucose ±5).
Calculates exact cosine similarity and mean absolute difference from live model
predictions without hardcoded numbers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from explainability.utils import (
    validate_patient_dict,
    patient_dict_to_dataframe,
    load_default_pipeline,
    load_preprocessor
)
from explainability.shap_explainer import explain_patient_tree_shap


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes cosine similarity between two feature attribution vectors.
    Returns 1.0 if vectors are identical, 0.0 if either norm is zero.
    """
    u = np.asarray(vec1, dtype=float).flatten()
    v = np.asarray(vec2, dtype=float).flatten()

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0.0

    dot_product = np.dot(u, v)
    cos_sim = dot_product / (norm_u * norm_v)
    # Clip numerical precision anomalies to [-1.0, 1.0]
    return float(np.clip(cos_sim, -1.0, 1.0))


def generate_physiological_perturbations(base_patient: dict) -> list[tuple[str, dict]]:
    """
    Constructs the 6 standard physiological perturbation variants used in the research:
    1. Age + 1 year
    2. Age - 1 year
    3. BMI + 1 kg/m²
    4. BMI - 1 kg/m²
    5. Blood Glucose + 5 mg/dL
    6. Blood Glucose - 5 mg/dL
    """
    clean = validate_patient_dict(base_patient)
    perturbations = []

    # 1. Age +1
    p1 = clean.copy()
    p1["age"] = float(p1["age"]) + 1.0
    perturbations.append(("Age +1 year", p1))

    # 2. Age -1
    p2 = clean.copy()
    p2["age"] = max(0.0, float(p2["age"]) - 1.0)
    perturbations.append(("Age -1 year", p2))

    # 3. BMI +1
    p3 = clean.copy()
    p3["bmi"] = float(p3["bmi"]) + 1.0
    perturbations.append(("BMI +1", p3))

    # 4. BMI -1
    p4 = clean.copy()
    p4["bmi"] = max(10.0, float(p4["bmi"]) - 1.0)
    perturbations.append(("BMI -1", p4))

    # 5. Glucose +5
    p5 = clean.copy()
    p5["avg_glucose_level"] = float(p5["avg_glucose_level"]) + 5.0
    perturbations.append(("Glucose +5", p5))

    # 6. Glucose -5
    p6 = clean.copy()
    p6["avg_glucose_level"] = max(40.0, float(p6["avg_glucose_level"]) - 5.0)
    perturbations.append(("Glucose -5", p6))

    return perturbations


def evaluate_explanation_stability(
    patient_data: dict | pd.DataFrame,
    pipeline=None,
    model_name: str = "tuned_xgboost",
    save_plot_path: str = None
) -> dict:
    """
    Computes explanation stability metrics for an individual patient under
    physiological perturbations.

    Parameters:
    - patient_data: Baseline patient dict or 1-row DataFrame.
    - pipeline: Optional pre-loaded pipeline.
    - model_name: Name of model pipeline to evaluate (default: 'tuned_xgboost').
    - save_plot_path: Optional path to save stability bar plot.

    Returns:
    - Dictionary with perturbation breakdown table, average cosine similarity,
      and average mean absolute SHAP shift.
    """
    if pipeline is None:
        pipeline = load_default_pipeline(model_name)

    if isinstance(patient_data, pd.DataFrame):
        patient_dict = patient_data.iloc[0].to_dict()
    else:
        patient_dict = validate_patient_dict(patient_data)

    # 1. Compute baseline explanation
    baseline_result = explain_patient_tree_shap(
        patient_data=patient_dict,
        pipeline=pipeline,
        model_name=model_name
    )
    baseline_prob = baseline_result["predicted_stroke_probability"]
    baseline_shap = np.asarray(baseline_result["raw_shap_vector"], dtype=float)

    # 2. Evaluate all perturbations
    perturbations = generate_physiological_perturbations(patient_dict)
    records = []

    for name, perturbed_dict in perturbations:
        pert_result = explain_patient_tree_shap(
            patient_data=perturbed_dict,
            pipeline=pipeline,
            model_name=model_name
        )
        pert_prob = pert_result["predicted_stroke_probability"]
        pert_shap = np.asarray(pert_result["raw_shap_vector"], dtype=float)

        cos_sim = calculate_cosine_similarity(baseline_shap, pert_shap)
        mad = float(np.mean(np.abs(baseline_shap - pert_shap)))

        records.append({
            "perturbation": name,
            "baseline_probability": round(baseline_prob, 4),
            "perturbed_probability": round(pert_prob, 4),
            "probability_shift": round(pert_prob - baseline_prob, 4),
            "cosine_similarity": round(cos_sim, 4),
            "mean_absolute_shap_difference": round(mad, 6)
        })

    df_stability = pd.DataFrame(records)
    avg_cosine = float(df_stability["cosine_similarity"].mean())
    avg_mad = float(df_stability["mean_absolute_shap_difference"].mean())

    summary = {
        "model_name": model_name,
        "baseline_probability": baseline_prob,
        "average_cosine_similarity": round(avg_cosine, 4),
        "average_mean_absolute_shap_difference": round(avg_mad, 6),
        "perturbation_details": records
    }

    # 3. Optional visualization
    if save_plot_path:
        plt.figure(figsize=(9, 5))
        plt.bar(df_stability["perturbation"], df_stability["cosine_similarity"], color="#337ab7")
        plt.axhline(avg_cosine, color="#d9534f", linestyle="--", label=f"Average Cosine Sim ({avg_cosine:.4f})")
        plt.xlabel("Input Perturbation")
        plt.ylabel("SHAP Cosine Similarity")
        plt.title(f"Explanation Stability Under Perturbations ({model_name})")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return summary
