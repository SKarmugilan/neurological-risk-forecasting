"""
explainability/lime_explainer.py - LIME Explanation Module
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

Provides local model-agnostic feature attributions using LIME (Local Interpretable
Model-agnostic Explanations) for individual models (Tuned XGBoost, Random Forest)
and the Stacking Ensemble.
"""

import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from explainability.utils import (
    patient_dict_to_dataframe,
    load_default_pipeline,
    load_preprocessor,
    get_feature_names_from_preprocessor,
    get_training_background,
    format_explanation_for_json,
    plot_feature_contributions
)


def create_lime_explainer(
    preprocessor=None,
    n_background_samples: int = 1000,
    random_state: int = 42
) -> LimeTabularExplainer:
    """
    Initializes a LimeTabularExplainer fitted on preprocessed training background samples.
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # Load preprocessed training data for empirical background distribution
    background_processed = get_training_background(
        n_samples=n_background_samples,
        random_state=random_state,
        preprocessed=True
    )
    if hasattr(background_processed, "toarray"):
        background_processed = background_processed.toarray()

    explainer = LimeTabularExplainer(
        training_data=background_processed,
        feature_names=feature_names,
        class_names=["No Stroke", "Stroke"],
        mode="classification",
        discretize_continuous=False,
        random_state=random_state
    )

    return explainer


def explain_patient_lime(
    patient_data: dict | pd.DataFrame,
    pipeline=None,
    model_name: str = "tuned_xgboost",
    num_features: int = 10,
    num_samples: int = 2000,
    random_state: int = 42,
    save_plot_path: str = None
) -> dict:
    """
    Generates a LIME local explanation for an individual patient.

    Parameters:
    - patient_data: Raw patient record dictionary or 1-row DataFrame.
    - pipeline: Optional pre-loaded model or pipeline. If None, loaded via model_name.
    - model_name: Name of pipeline ('tuned_xgboost', 'stacking_ensemble', 'random_forest', etc.).
    - num_features: Number of top features to include in the linear surrogate model.
    - num_samples: Number of perturbation samples drawn by LIME.
    - random_state: Random seed for perturbation reproducibility.
    - save_plot_path: Optional path to save bar plot.

    Returns:
    - Dictionary with LIME explanation metadata, ranked feature weights, and predicted probability.
    """
    if pipeline is None:
        pipeline = load_default_pipeline(model_name)

    preprocessor = load_preprocessor()
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # 1. Prepare raw patient input
    if isinstance(patient_data, dict):
        df_patient = patient_dict_to_dataframe(patient_data)
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy()
    else:
        raise TypeError(f"Expected dict or DataFrame, got {type(patient_data).__name__}")

    # 2. Transform patient features
    patient_proc = preprocessor.transform(df_patient)
    if hasattr(patient_proc, "toarray"):
        patient_proc = patient_proc.toarray()
    patient_vec = patient_proc[0]

    # 3. Setup prediction function based on model architecture
    if model_name == "stacking_ensemble" or hasattr(pipeline, "final_estimator_"):
        # For StackingClassifier, evaluate meta-model from preprocessed array
        def predict_fn(X_proc_array):
            meta_features = np.column_stack([
                est.named_steps["model"].predict_proba(X_proc_array)[:, 1]
                for est in pipeline.estimators_
            ])
            return pipeline.final_estimator_.predict_proba(meta_features)

        prob_stroke = float(pipeline.predict_proba(df_patient)[0][1])
        model_label = "StackingClassifier Ensemble"
    else:
        # For Pipeline with ('preprocessing', preprocessor) and ('model', estimator)
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            estimator = pipeline.named_steps["model"]
            predict_fn = estimator.predict_proba
            prob_stroke = float(estimator.predict_proba(patient_proc)[0][1])
            model_label = estimator.__class__.__name__
        else:
            predict_fn = pipeline.predict_proba
            prob_stroke = float(pipeline.predict_proba(df_patient)[0][1])
            model_label = pipeline.__class__.__name__

    # 4. Create LIME explainer and explain instance
    explainer = create_lime_explainer(
        preprocessor=preprocessor,
        n_background_samples=1000,
        random_state=random_state
    )

    explanation = explainer.explain_instance(
        data_row=patient_vec,
        predict_fn=predict_fn,
        labels=(1,),
        num_features=min(num_features, len(feature_names)),
        num_samples=num_samples
    )

    # 5. Extract weights for Stroke class (label 1)
    lime_tuples = explanation.as_list(label=1)

    # Construct complete contribution vector for consistency
    weights_dict = dict(lime_tuples)
    full_weights = [weights_dict.get(feat, 0.0) for feat in feature_names]

    formatted = format_explanation_for_json(
        feature_names=feature_names,
        contributions=full_weights,
        prediction_prob=prob_stroke
    )

    # Filter to requested top features
    top_features = formatted["ranked_features"][:num_features]

    result = {
        "explanation_method": "LIME (Local Interpretable Model-agnostic Explanations)",
        "model_explained": model_label,
        "pipeline_name": model_name,
        "predicted_stroke_probability": prob_stroke,
        "local_score_intercept": float(explanation.intercept[1]) if hasattr(explanation, "intercept") and 1 in explanation.intercept else None,
        "local_score_r2": float(explanation.score) if hasattr(explanation, "score") else None,
        "top_feature_contributions": top_features,
        "all_feature_contributions": formatted["ranked_features"]
    }

    if save_plot_path:
        plot_feature_contributions(
            top_features,
            title=f"LIME Local Explanation ({model_label})",
            save_path=save_plot_path
        )

    return result
