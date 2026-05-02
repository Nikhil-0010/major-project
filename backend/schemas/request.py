# backend/schemas/request.py
"""
Pydantic input schema for /predict endpoint.

Validates and documents all 13 patient features.
ECG-related fields (oldpeak, exang, slope, thal) are optional
to support fallback to XGB-only mode.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Patient clinical data for heart disease risk prediction."""

    age: int = Field(..., ge=20, le=100, description="Patient age in years")

    sex: Literal["male", "female"] = Field(
        ..., description="Biological sex"
    )

    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = Field(
        ..., description="Chest pain type"
    )

    trestbps: int = Field(
        ..., ge=80, le=200, description="Resting blood pressure (mmHg)"
    )

    chol: int = Field(
        ..., ge=100, le=600, description="Serum cholesterol (mg/dL)"
    )

    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dL (1=true, 0=false)"
    )

    restecg: Literal["normal", "st-t abnormality", "lv hypertrophy"] = Field(
        ..., description="Resting ECG result"
    )

    thalch: int = Field(
        ..., ge=60, le=220, description="Maximum heart rate achieved (bpm)"
    )

    # --- ECG-related fields (optional for fallback) ---

    exang: Optional[int] = Field(
        None, ge=0, le=1,
        description="Exercise-induced angina (1=yes, 0=no). Omit for XGB-only mode."
    )

    oldpeak: Optional[float] = Field(
        None, ge=0.0, le=6.2,
        description="ST depression induced by exercise (mm). Omit for XGB-only mode."
    )

    slope: Optional[Literal["upsloping", "flat", "downsloping"]] = Field(
        None, description="Slope of peak exercise ST segment. Omit for XGB-only mode."
    )

    thal: Optional[Literal["normal", "fixed defect", "reversable defect"]] = Field(
        None, description="Thallium stress test result. Omit for XGB-only mode."
    )

    ca: int = Field(
        ..., ge=0, le=4,
        description="Number of major vessels colored by fluoroscopy (0-4)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63,
                    "sex": "male",
                    "cp": "asymptomatic",
                    "trestbps": 145,
                    "chol": 250,
                    "fbs": 0,
                    "restecg": "normal",
                    "thalch": 150,
                    "exang": 1,
                    "oldpeak": 2.3,
                    "slope": "flat",
                    "ca": 2,
                    "thal": "reversable defect",
                }
            ]
        }
    }
