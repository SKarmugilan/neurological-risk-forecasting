"""
explainability/shap_explainer.py - SHAP Explanation Module
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

Provides TreeSHAP explanations for tree-based models (Tuned XGBoost, Random Forest)
and Model-Agnostic KernelSHAP explanations for the heterogeneous Stacking Ensemble.

IMPORTANT METHODOLOGICAL NOTE:
TreeSHAP utilizes exact polynomial-time tree path traversals and is valid ONLY for tree-based
estimators (e.g. XGBoost, Random Forest). It CANNOT be executed directly on the heterogeneous
StackingClassifier (which stacks linear and tree models through a logistic regression meta-learner).
Explanations for the Stacking Ensemble utilize Model-Agnostic KernelSHAP and are explicitly
documented as such.
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from explainability.utils import (
    patient_dict_to_dataframe,
    load_default_pipeline,
    load_preprocessor,
    get_feature_names_from_preprocessor,
    get_training_background,
    format_explanation_for_json,
    plot_feature_contributions
)


def explain_patient_tree_shap(
    patient_data: dict | pd.DataFrame,
    pipeline=None,
    model_name: str = "tuned_xgboost",
    save_plot_path: str = None
) -> dict:
    """
    Computes local TreeSHAP feature attributions for a single patient instance
    using a tree-based model (e.g. Tuned XGBoost or Random Forest).

    Parameters:
    - patient_data: Raw patient dict or 1-row DataFrame matching the 10 raw features.
    - pipeline: Optional pre-loaded Pipeline. If None, loaded via model_name.
    - model_name: Name of pipeline to load ('tuned_xgboost', 'xgboost', 'random_forest').
    - save_plot_path: Optional path to save horizontal bar chart.

    Returns:
    - Dictionary with explanation metadata, ranked feature attributions, base value, and probability.
    """
    if pipeline is None:
        pipeline = load_default_pipeline(model_name)

    # 1. Prepare raw input DataFrame
    if isinstance(patient_data, dict):
        df_patient = patient_dict_to_dataframe(patient_data)
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(patient_data).__name__}")

    # 2. Extract preprocessor and underlying tree model
    if hasattr(pipeline, "named_steps"):
        preprocessor = pipeline.named_steps.get("preprocessing")
        tree_model = pipeline.named_steps.get("model")
    else:
        preprocessor = load_preprocessor()
        tree_model = pipeline

    if preprocessor is None:
        preprocessor = load_preprocessor()

    # 3. Transform patient features
    patient_processed = preprocessor.transform(df_patient)
    if hasattr(patient_processed, "toarray"):
        patient_processed = patient_processed.toarray()

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # 4. Predict probability for stroke class (index 1)
    if hasattr(tree_model, "predict_proba"):
        prob_stroke = float(tree_model.predict_proba(patient_processed)[0][1])
    else:
        prob_stroke = float(pipeline.predict_proba(df_patient)[0][1])

    # 5. TreeSHAP computation
    explainer = shap.TreeExplainer(tree_model)
    shap_output = explainer(patient_processed)

    # Extract SHAP values for positive class (Stroke)
    if len(shap_output.values.shape) == 3:  # (n_samples, n_features, n_classes) e.g. Random Forest
        shap_values = shap_output.values[0, :, 1]
        base_value = float(shap_output.base_values[0, 1])
    elif len(shap_output.values.shape) == 2:  # (n_samples, n_features) e.g. XGBoost
        shap_values = shap_output.values[0, :]
        base_val = shap_output.base_values[0]
        base_value = float(base_val[1]) if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1 else float(base_val)
    else:
        shap_values = np.asarray(shap_output.values).flatten()
        base_value = float(shap_output.base_values[0]) if hasattr(shap_output, "base_values") else 0.0

    # 6. Format formatted JSON-compatible output
    formatted = format_explanation_for_json(
        feature_names=feature_names,
        contributions=shap_values,
        base_value=base_value,
        prediction_prob=prob_stroke
    )

    result = {
        "explanation_method": "TreeSHAP",
        "model_explained": tree_model.__class__.__name__,
        "pipeline_name": model_name,
        "predicted_stroke_probability": prob_stroke,
        "base_value": base_value,
        "feature_contributions": formatted["ranked_features"],
        "raw_shap_vector": shap_values.tolist()
    }

    # 7. Optional plot generation
    if save_plot_path:
        plot_feature_contributions(
            formatted["ranked_features"],
            title=f"TreeSHAP Explanation ({tree_model.__class__.__name__})",
            save_path=save_plot_path
        )

    return result


def explain_ensemble_kernel_shap(
    patient_data: dict | pd.DataFrame,
    ensemble_pipeline=None,
    background_samples: int = 30,
    nsamples: int = 100,
    save_plot_path: str = None
) -> dict:
    """
    Computes Model-Agnostic KernelSHAP feature attributions for the Stacking Ensemble.

    IMPORTANT:
    This is an empirical model-agnostic explanation (KernelSHAP) using black-box output
    probabilities over the transformed feature space. It is computationally heavier than
    TreeSHAP. For interactive web applications, background_samples=25-30 and nsamples=100
    provide a fast response time (<0.5s) while accurately estimating feature contributions.

    Parameters:
    - patient_data: Raw patient dict or 1-row DataFrame.
    - ensemble_pipeline: Optional pre-loaded StackingClassifier.
    - background_samples: Number of background references sampled from X_train.
    - nsamples: Number of perturbation evaluations for KernelSHAP.
    - save_plot_path: Optional path to save bar plot.

    Returns:
    - Dictionary with model-agnostic explanation results and ranked contributions.
    """
    if ensemble_pipeline is None:
        ensemble_pipeline = load_default_pipeline("stacking_ensemble")

    preprocessor = load_preprocessor()
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # 1. Prepare raw patient input
    if isinstance(patient_data, dict):
        df_patient = patient_dict_to_dataframe(patient_data)
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(patient_data).__name__}")

    # 2. Transform patient instance
    patient_proc = preprocessor.transform(df_patient)
    if hasattr(patient_proc, "toarray"):
        patient_proc = patient_proc.toarray()

    # 3. Predict stroke probability directly from ensemble
    prob_stroke = float(ensemble_pipeline.predict_proba(df_patient)[0][1])

    # 4. Prepare preprocessed prediction function for StackingClassifier
    # StackingClassifier meta-model evaluates base estimator models from preprocessed features
    def ensemble_preprocessed_predict_fn(X_proc_array):
        # Base estimators in StackingClassifier each have a 'model' step
        meta_features = np.column_stack([
            est.named_steps["model"].predict_proba(X_proc_array)[:, 1]
            for est in ensemble_pipeline.estimators_
        ])
        return ensemble_pipeline.final_estimator_.predict_proba(meta_features)[:, 1]

    # 5. Background reference data
    background_proc = get_training_background(
        n_samples=background_samples,
        random_state=42,
        preprocessed=True
    )
    if hasattr(background_proc, "toarray"):
        background_proc = background_proc.toarray()

    # 6. KernelSHAP computation
    explainer = shap.KernelExplainer(
        ensemble_preprocessed_predict_fn,
        background_proc
    )

    shap_values = explainer.shap_values(
        patient_proc,
        nsamples=nsamples,
        silent=True
    )

    if isinstance(shap_values, list):
        shap_vec = np.asarray(shap_values[0]).flatten()
    else:
        shap_vec = np.asarray(shap_values).flatten()

    base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    # 7. Format results
    formatted = format_explanation_for_json(
        feature_names=feature_names,
        contributions=shap_vec,
        base_value=base_val,
        prediction_prob=prob_stroke
    )

    result = {
        "explanation_method": "Model-Agnostic KernelSHAP",
        "model_explained": "StackingClassifier Ensemble",
        "pipeline_name": "stacking_ensemble",
        "predicted_stroke_probability": prob_stroke,
        "base_value": base_val,
        "feature_contributions": formatted["ranked_features"],
        "raw_shap_vector": shap_vec.tolist(),
        "computational_settings": {
            "background_samples": background_samples,
            "nsamples": nsamples
        }
    }

    if save_plot_path:
        plot_feature_contributions(
            formatted["ranked_features"],
            title="Model-Agnostic KernelSHAP (Stacking Ensemble)",
            save_path=save_plot_path
        )

    return result


def generate_global_tree_shap(
    pipeline=None,
    model_name: str = "tuned_xgboost",
    max_samples: int = 500,
    save_plot_path: str = None
) -> pd.DataFrame:
    """
    Computes global feature importance rankings using TreeSHAP over a sample of the cohort.
    """
    if pipeline is None:
        pipeline = load_default_pipeline(model_name)

    preprocessor = pipeline.named_steps.get("preprocessing", load_preprocessor())
    tree_model = pipeline.named_steps.get("model", pipeline)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    X_sample_proc = get_training_background(n_samples=max_samples, preprocessed=True)
    if hasattr(X_sample_proc, "toarray"):
        X_sample_proc = X_sample_proc.toarray()

    explainer = shap.TreeExplainer(tree_model)
    shap_vals = explainer.shap_values(X_sample_proc)

    if isinstance(shap_vals, list):
        shap_matrix = shap_vals[1]
    else:
        shap_matrix = shap_vals

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    if save_plot_path:
        plt.figure(figsize=(10, 7))
        top_df = df_importance.head(15).iloc[::-1]
        plt.barh(top_df["Feature"], top_df["Mean_Absolute_SHAP"], color="#337ab7")
        plt.xlabel("Mean Absolute SHAP Value")
        plt.title(f"Global Feature Importance (TreeSHAP - {tree_model.__class__.__name__})")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_plot_path)), exist_ok=True)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return df_importance
