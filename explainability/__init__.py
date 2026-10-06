"""
Explainable AI (XAI) Package
Project: Explainable AI on Ensemble Methods for Early Neurological Risk Forecasting

This package provides clean, modular, and reusable interpretability tools for stroke
risk prediction models, supporting both tree-based models (Tuned XGBoost) and the
heterogeneous Stacking Ensemble.
"""

from explainability.utils import (
    validate_patient_dict,
    patient_dict_to_dataframe,
    load_saved_pipeline,
    load_default_pipeline,
    load_preprocessor,
    get_feature_names_from_preprocessor,
    format_explanation_for_json,
    plot_feature_contributions,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXPECTED_RAW_FEATURES
)

from explainability.shap_explainer import (
    explain_patient_tree_shap,
    explain_ensemble_kernel_shap,
    generate_global_tree_shap
)

from explainability.lime_explainer import (
    explain_patient_lime,
    create_lime_explainer
)

from explainability.stability import (
    evaluate_explanation_stability,
    calculate_cosine_similarity,
    generate_physiological_perturbations
)

__all__ = [
    "validate_patient_dict",
    "patient_dict_to_dataframe",
    "load_saved_pipeline",
    "load_default_pipeline",
    "load_preprocessor",
    "get_feature_names_from_preprocessor",
    "format_explanation_for_json",
    "plot_feature_contributions",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "EXPECTED_RAW_FEATURES",
    "explain_patient_tree_shap",
    "explain_ensemble_kernel_shap",
    "generate_global_tree_shap",
    "explain_patient_lime",
    "create_lime_explainer",
    "evaluate_explanation_stability",
    "calculate_cosine_similarity",
    "generate_physiological_perturbations"
]
