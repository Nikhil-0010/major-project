# backend/api/predict.py
"""
POST /predict endpoint.

Accepts patient data, runs prediction pipeline,
generates clinical explanation, and returns structured response.
"""

from fastapi import APIRouter, HTTPException
import traceback

from backend.schemas.request import PatientInput
from backend.schemas.response import (
    PredictionResponse,
    FactorItem,
    Recommendation,
    SourceInfo,
    ModelInfo,
)
from backend.services.prediction import predict
from backend.services.explanation import build_explanation
from backend.models.load_models import registry

router = APIRouter()


def _factor_to_item(factor: dict) -> FactorItem:
    """Convert raw factor dict from clinical module to response schema."""
    return FactorItem(
        feature=factor.get("feature", ""),
        value=factor.get("value", ""),
        severity=factor.get("severity", ""),
        label=factor.get("label", ""),
        shap_impact=factor.get("shap_impact", 0.0),
        impact_strength=factor.get("impact_strength", ""),
        doctor_explanation=factor.get("doctor_explanation", ""),
        patient_explanation=factor.get("patient_explanation", ""),
        source=factor.get("source", ""),
        clinical_type=factor.get("clinical_type", ""),
        muted=factor.get("muted", False),
        muted_note=factor.get("muted_note", ""),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(patient: PatientInput):
    """
    Predict heart disease risk from patient clinical data.

    Returns a comprehensive response with:
    - Risk probability and label
    - XGB/ECG/hybrid probabilities
    - SHAP feature contributions
    - Doctor and patient explanations
    - Clinical recommendations
    - Medical reference sources
    """
    try:
        # Convert Pydantic model to dict for processing
        raw_input = patient.model_dump()

        # --- Run prediction pipeline ---
        prediction = predict(raw_input)

        # --- Generate clinical explanation (uses RAW input) ---
        explanation = build_explanation(raw_input, prediction)

        # --- Build response ---
        risk_label = "HIGH RISK" if prediction.final_prob >= 0.7 else (
            "MODERATE RISK" if prediction.final_prob >= 0.4 else "LOW RISK"
        )

        # SHAP data for frontend charting
        shap_features = prediction.shap_feature_names
        shap_values = [float(v) for v in prediction.shap_values]

        # Build factor items
        risk_factors = [_factor_to_item(f) for f in explanation["risk_factors"]]
        protective_factors = [_factor_to_item(f) for f in explanation["protective_factors"]]

        # Build recommendations
        rec_doc = explanation["recommendation_doctor"]
        rec_pat = explanation["recommendation_patient"]

        recommendation_doctor = Recommendation(
            risk_level=rec_doc["risk_level"],
            urgency=rec_doc["urgency"],
            action=rec_doc["action"],
            detail=rec_doc["detail"],
            key_focus_areas=rec_doc.get("key_focus_areas", []),
            mode=rec_doc.get("mode", "doctor"),
        )

        recommendation_patient = Recommendation(
            risk_level=rec_pat["risk_level"],
            urgency=rec_pat["urgency"],
            action=rec_pat["action"],
            detail=rec_pat["detail"],
            key_focus_areas=rec_pat.get("key_focus_areas", []),
            mode=rec_pat.get("mode", "patient"),
        )

        # Build sources
        sources = [
            SourceInfo(
                name=s["name"],
                citation=s["citation"],
                description=s["description"],
            )
            for s in explanation["sources"]
        ]

        # Model info
        model_info = ModelInfo(
            xgb_auc=registry.xgb_auc,
            hybrid_auc=registry.hybrid_auc,
        )

        return PredictionResponse(
            probability=round(prediction.final_prob, 4),
            risk_label=risk_label,
            mode=prediction.mode,
            xgb_prob=round(prediction.xgb_prob, 4),
            ecg_prob=round(prediction.ecg_prob, 4) if prediction.ecg_prob is not None else None,
            doctor_view=explanation["doctor_view"],
            patient_view=explanation["patient_view"],
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            shap_features=shap_features,
            shap_values=shap_values,
            recommendation_doctor=recommendation_doctor,
            recommendation_patient=recommendation_patient,
            sources=sources,
            model_info=model_info,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
