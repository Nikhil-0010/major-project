# backend/services/prediction.py
"""
Core prediction pipeline.

Handles:
1. Input normalization (categorical formats + booleans)
2. Preprocessing via fitted ColumnTransformer
3. XGB stream prediction (top-7 selected features)
4. ECG stream prediction (if ECG features available)
5. Meta-model fusion (hybrid mode)
6. Fallback to XGB-only when ECG features missing
7. SHAP value computation on selected features

IMPORTANT:
- Never manually scale numerics — the preprocessor does that
- Only fix categorical formats to match training data
- SHAP is computed on X_sel (selected features), not full X_trans
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple

from backend.models.load_models import registry


# ECG fields that must ALL be present for hybrid mode
ECG_FIELDS = ["oldpeak", "exang", "slope", "thal"]


@dataclass
class PredictionResult:
    """Holds all outputs from the prediction pipeline."""
    xgb_prob: float
    ecg_prob: Optional[float]
    final_prob: float
    mode: str  # "hybrid" or "xgb_only"
    shap_values: np.ndarray  # SHAP values for selected features
    shap_feature_names: List[str]  # names of selected features


def _normalize_input(data: dict) -> dict:
    """
    Normalize categorical values and booleans to match training format.

    Training data used:
    - sex: "Male" / "Female" (capitalized)
    - fbs: "True" / "False" (string booleans)
    - exang: "True" / "False" (string booleans)
    - cp, slope, thal, restecg: lowercase strings (already match)

    Frontend sends:
    - sex: "male" / "female"
    - fbs: 0 / 1
    - exang: 0 / 1 / None
    """
    normalized = data.copy()

    # Sex: "male" → "Male"
    if "sex" in normalized and normalized["sex"] is not None:
        normalized["sex"] = str(normalized["sex"]).capitalize()

    # FBS: 0/1 → "True"/"False" string
    if "fbs" in normalized:
        normalized["fbs"] = str(bool(normalized["fbs"]))

    # Exang: 0/1/None → "True"/"False"/None string
    if "exang" in normalized and normalized["exang"] is not None:
        normalized["exang"] = str(bool(normalized["exang"]))

    return normalized


def _check_ecg_available(data: dict) -> bool:
    """Check if ALL ECG features are present and non-None."""
    return all(
        field in data and data[field] is not None
        for field in ECG_FIELDS
    )


def predict(input_data: dict) -> PredictionResult:
    """
    Run the full prediction pipeline on a single patient.

    Args:
        input_data: Raw patient data dict (from API request)

    Returns:
        PredictionResult with probabilities, mode, and SHAP values
    """
    reg = registry

    # 1. Normalize categorical formats (DO NOT touch numerics)
    normalized = _normalize_input(input_data)

    # 2. Check ECG availability BEFORE normalization changes anything
    ecg_available = _check_ecg_available(input_data)

    # 3. Build DataFrame (single row)
    # For missing ECG fields, fill with defaults so preprocessor doesn't crash
    if not ecg_available:
        for field in ECG_FIELDS:
            if field not in normalized or normalized[field] is None:
                if field == "oldpeak":
                    normalized[field] = 0.0
                elif field == "exang":
                    normalized[field] = "False"
                elif field == "slope":
                    normalized[field] = "flat"
                elif field == "thal":
                    normalized[field] = "normal"

    X = pd.DataFrame([normalized])

    # 4. Transform through the fitted preprocessor
    X_trans = reg.preprocessor.transform(X)

    # 5. XGB stream prediction (selected features only)
    X_xgb = X_trans[:, reg.xgb_indices]
    xgb_prob = float(reg.xgb_model.predict_proba(X_xgb)[:, 1][0])

    # 6. ECG stream + meta-model (if available)
    ecg_prob = None
    if ecg_available:
        X_ecg = X_trans[:, reg.ecg_indices]
        ecg_prob = float(reg.ecg_model.predict_proba(X_ecg)[:, 1][0])

        # Meta-model fusion
        meta_input = np.array([[xgb_prob, ecg_prob]])
        final_prob = float(reg.meta_model.predict_proba(meta_input)[:, 1][0])
        mode = "hybrid"
    else:
        final_prob = xgb_prob
        mode = "xgb_only"

    # 7. SHAP values (on XGB selected features)
    shap_vals = reg.explainer.shap_values(X_xgb)

    # Handle different SHAP output formats
    if isinstance(shap_vals, list):
        # Binary classification: [class_0, class_1]
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    shap_values_1d = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    # Feature names for the selected features
    shap_feature_names = [reg.transformed_feature_names[i] for i in reg.xgb_indices]

    return PredictionResult(
        xgb_prob=xgb_prob,
        ecg_prob=ecg_prob,
        final_prob=final_prob,
        mode=mode,
        shap_values=shap_values_1d,
        shap_feature_names=shap_feature_names,
    )
