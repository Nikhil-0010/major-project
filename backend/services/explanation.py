# backend/services/explanation.py
"""
Bridge between the prediction pipeline and existing clinical logic.

Calls into src/clinical/* WITHOUT duplicating any code:
- generate_explanation() → risk/protective factors + dual views
- get_recommendation() → action + urgency for both audiences
- SOURCES → medical references for the frontend

IMPORTANT: Clinical explanation expects RAW patient values (not transformed).
"""

from src.clinical.explanation import generate_explanation
from src.clinical.recommendation import get_recommendation
from src.clinical.sources import SOURCES

from backend.services.prediction import PredictionResult


def build_explanation(raw_input: dict, prediction: PredictionResult) -> dict:
    """
    Generate full clinical explanation from raw patient data + prediction results.

    Args:
        raw_input: Original patient data (pre-normalization, raw values)
        prediction: PredictionResult from the prediction pipeline

    Returns:
        dict with:
            - risk_factors, protective_factors
            - doctor_view, patient_view
            - recommendation_doctor, recommendation_patient
            - sources (list of medical references)
    """
    # 1. Generate explanation using RAW patient values + SHAP
    explanation = generate_explanation(
        patient_row=raw_input,
        shap_values=prediction.shap_values,
        feature_names=prediction.shap_feature_names,
        prob=prediction.final_prob,
        top_k=5,
    )

    # 2. Generate recommendations for BOTH audiences
    recommendation_doctor = get_recommendation(explanation, mode="doctor")
    recommendation_patient = get_recommendation(explanation, mode="patient")

    # 3. Collect relevant sources
    sources = []
    seen = set()
    # Add sources for all features that appeared in factors
    all_factors = explanation.get("risk_factors", []) + explanation.get("protective_factors", [])
    for factor in all_factors:
        feature = factor.get("feature", "")
        if feature in SOURCES and feature not in seen:
            seen.add(feature)
            src = SOURCES[feature]
            sources.append({
                "name": src["name"],
                "citation": src["citation"],
                "description": src["description"],
                "url": src.get("url"),
            })

    # Always include key guideline sources
    for key in ["oldpeak", "chol", "trestbps", "fbs"]:
        if key in SOURCES and key not in seen:
            seen.add(key)
            src = SOURCES[key]
            sources.append({
                "name": src["name"],
                "citation": src["citation"],
                "description": src["description"],
                "url": src.get("url"),
            })

    return {
        "risk_factors": explanation.get("risk_factors", []),
        "protective_factors": explanation.get("protective_factors", []),
        "doctor_view": explanation.get("doctor_view", ""),
        "patient_view": explanation.get("patient_view", ""),
        "recommendation_doctor": recommendation_doctor,
        "recommendation_patient": recommendation_patient,
        "sources": sources,
    }
