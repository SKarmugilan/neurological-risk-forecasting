from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explainability.shap_explainer import explain_ensemble_kernel_shap


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "stacking_ensemble_pipeline.joblib"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Neurological Risk Forecasting API",
    description=(
        "Research prototype API for stroke-risk forecasting "
        "using a stacking ensemble with explainable AI."
    ),
    version="1.0.0",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# PATIENT INPUT MODEL
# =========================================================

class PatientData(BaseModel):

    age: float

    gender: str

    hypertension: int

    heart_disease: int

    ever_married: str

    work_type: str

    Residence_type: str

    avg_glucose_level: float

    bmi: float

    smoking_status: str


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message":
            "Neurological Risk Forecasting API is running",

        "model":
            "Stacking Ensemble",

        "status":
            "ready",

        "explainability":
            "KernelSHAP",

        "disclaimer":
            (
                "This is a research prototype and is not a "
                "clinically validated diagnostic tool."
            ),
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model_loaded":
            True,

        "explainability_loaded":
            True,
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(patient: PatientData):

    # -----------------------------------------------------
    # Convert patient data into DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "age":
                    patient.age,

                "gender":
                    patient.gender,

                "hypertension":
                    patient.hypertension,

                "heart_disease":
                    patient.heart_disease,

                "ever_married":
                    patient.ever_married,

                "work_type":
                    patient.work_type,

                "Residence_type":
                    patient.Residence_type,

                "avg_glucose_level":
                    patient.avg_glucose_level,

                "bmi":
                    patient.bmi,

                "smoking_status":
                    patient.smoking_status,
            }
        ]
    )


    # -----------------------------------------------------
    # Get model probability
    # -----------------------------------------------------

    probability = float(
        model.predict_proba(input_data)[0][1]
    )


    # -----------------------------------------------------
    # Research decision threshold
    # -----------------------------------------------------

    threshold = 0.15


    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # -----------------------------------------------------
    # Determine risk level
    # -----------------------------------------------------

    if prediction == 1:

        risk_level = "Elevated"

    else:

        risk_level = "Lower"


    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {

        "prediction":
            prediction,

        "stroke_probability":
            round(probability, 4),

        "stroke_probability_percent":
            round(probability * 100, 2),

        "threshold":
            threshold,

        "risk_level":
            risk_level,

        "model":
            "Stacking Ensemble",

        "disclaimer":
            (
                "This result is from a research prototype "
                "and must not be used as a medical diagnosis."
            ),
    }


# =========================================================
# EXPLAINABILITY ENDPOINT
# =========================================================

@app.post("/explain")
def explain(patient: PatientData):

    # -----------------------------------------------------
    # Convert Pydantic object into dictionary
    # -----------------------------------------------------

    patient_data = {

        "age":
            patient.age,

        "gender":
            patient.gender,

        "hypertension":
            patient.hypertension,

        "heart_disease":
            patient.heart_disease,

        "ever_married":
            patient.ever_married,

        "work_type":
            patient.work_type,

        "Residence_type":
            patient.Residence_type,

        "avg_glucose_level":
            patient.avg_glucose_level,

        "bmi":
            patient.bmi,

        "smoking_status":
            patient.smoking_status,
    }


    # -----------------------------------------------------
    # Generate KernelSHAP explanation
    # -----------------------------------------------------

    explanation = explain_ensemble_kernel_shap(

        patient_data,

        ensemble_pipeline=model,

        background_samples=10,

        nsamples=50,
    )


    # -----------------------------------------------------
    # Get probability
    # -----------------------------------------------------

    probability = float(
        explanation[
            "predicted_stroke_probability"
        ]
    )


    # -----------------------------------------------------
    # Decision threshold
    # -----------------------------------------------------

    threshold = 0.15


    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # -----------------------------------------------------
    # Determine risk level
    # -----------------------------------------------------

    if prediction == 1:

        risk_level = "Elevated"

    else:

        risk_level = "Lower"


    # -----------------------------------------------------
    # Return explanation
    # -----------------------------------------------------

    return {

        "prediction":
            prediction,

        "stroke_probability":
            round(probability, 4),

        "stroke_probability_percent":
            round(probability * 100, 2),

        "threshold":
            threshold,

        "risk_level":
            risk_level,

        "explanation_method":
            explanation[
                "explanation_method"
            ],

        "model":
            explanation[
                "model_explained"
            ],

        "base_value":
            explanation[
                "base_value"
            ],

        "top_features":
            explanation[
                "feature_contributions"
            ][:10],

        "disclaimer":
            (
                "This explanation is generated by a "
                "research prototype and must not be used "
                "as a medical diagnosis."
            ),
    }