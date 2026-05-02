# backend/schemas/response.py
"""
Pydantic response schema for /predict endpoint.

Structured to support both doctor and patient views,
SHAP-based feature contributions, and clinical recommendations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class FactorItem(BaseModel):
    """A single risk or protective factor."""
    feature: str
    value: Any
    severity: str
    label: str = ""
    shap_impact: float
    impact_strength: str
    doctor_explanation: str
    patient_explanation: str
    source: str = ""
    clinical_type: str = ""
    muted: bool = False
    muted_note: str = ""


class Recommendation(BaseModel):
    """Clinical recommendation based on risk level."""
    risk_level: str
    urgency: str
    action: str
    detail: str
    key_focus_areas: List[str] = []
    mode: str


class SourceInfo(BaseModel):
    """Medical reference source."""
    name: str
    citation: str
    description: str
    url: Optional[str] = None


class ModelInfo(BaseModel):
    """Model performance metrics."""
    xgb_auc: Optional[float] = None
    hybrid_auc: Optional[float] = None


class PredictionResponse(BaseModel):
    """Full prediction response with explanations."""

    # Core prediction
    probability: float
    risk_label: str
    mode: str  # "hybrid" or "xgb_only"

    # Stream probabilities
    xgb_prob: float
    ecg_prob: Optional[float] = None

    # Dual-audience views
    doctor_view: str
    patient_view: str

    # Structured factors
    risk_factors: List[FactorItem]
    protective_factors: List[FactorItem]

    # SHAP data for frontend charting
    shap_features: List[str] = []
    shap_values: List[float] = []

    # Recommendations
    recommendation_doctor: Recommendation
    recommendation_patient: Recommendation

    # Evidence sources
    sources: List[SourceInfo] = []

    # Model performance
    model_info: ModelInfo
