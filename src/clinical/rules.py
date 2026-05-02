"""
rules.py

Clinical threshold rules for cardiovascular risk interpretation.

Each function:
- Takes raw patient value
- Returns structured clinical interpretation
- Uses evidence-based thresholds (linked via sources.py)
"""

from src.clinical.sources import SOURCES
import numpy as np


# ----------------------------
# Helper structure
# ----------------------------
def build_output(feature, value, severity, label, threshold):
    return {
        "feature": feature,
        "value": float(value) if isinstance(value, (int, float, np.number)) else value,
        "severity": severity,
        "label": label,
        "threshold": threshold,
        "source": SOURCES.get(feature, {})
    }


# ----------------------------
# OLDPEAK (ST depression)
# ----------------------------
def rule_oldpeak(value):
    if value >= 2.0:
        return build_output(
            "oldpeak", value, "severe",
            "Significant ST depression (high ischemic risk)",
            "≥ 2.0 mm"
        )
    elif value >= 1.0:
        return build_output(
            "oldpeak", value, "moderate",
            "Moderate ST depression",
            "1.0 – 1.9 mm"
        )
    elif value > 0:
        return build_output(
            "oldpeak", value, "mild",
            "Mild ST depression",
            "0.1 – 0.9 mm"
        )
    else:
        return build_output(
            "oldpeak", value, "normal",
            "No ST depression",
            "0 mm"
        )


# ----------------------------
# CHOLESTEROL
# ----------------------------
def rule_chol(value):
    if value >= 240:
        return build_output("chol", value, "severe", "High cholesterol", "≥ 240 mg/dL")
    elif value >= 200:
        return build_output("chol", value, "moderate", "Borderline high cholesterol", "200–239 mg/dL")
    else:
        return build_output("chol", value, "normal", "Desirable cholesterol", "< 200 mg/dL")


# ----------------------------
# BLOOD PRESSURE
# ----------------------------
def rule_trestbps(value):
    if value >= 140:
        return build_output("trestbps", value, "severe", "Hypertension stage 2", "≥ 140 mmHg")
    elif value >= 130:
        return build_output("trestbps", value, "moderate", "Hypertension stage 1", "130–139 mmHg")
    elif value >= 120:
        return build_output("trestbps", value, "mild", "Elevated blood pressure", "120–129 mmHg")
    else:
        return build_output("trestbps", value, "normal", "Normal blood pressure", "< 120 mmHg")


# ----------------------------
# AGE
# ----------------------------
def rule_age(value):
    if value >= 65:
        return build_output("age", value, "severe", "Advanced age risk", "≥ 65 years")
    elif value >= 50:
        return build_output("age", value, "moderate", "Elevated age risk", "50–64 years")
    else:
        return build_output("age", value, "normal", "Lower age-related risk", "< 50 years")


# ----------------------------
# CA (vessels)
# ----------------------------
def rule_ca(value):
    if value >= 2:
        return build_output("ca", value, "severe", "Multiple vessel blockage", "≥ 2 vessels")
    elif value == 1:
        return build_output("ca", value, "moderate", "Single vessel involvement", "1 vessel")
    else:
        return build_output("ca", value, "normal", "No vessel blockage", "0 vessels")



# ----------------------------
# EXANG (exercise angina)
# ----------------------------
def rule_exang(value):
    if value == 1:
        return build_output("exang", value, "severe", "Exercise-induced angina present", "Yes")
    else:
        return build_output("exang", value, "normal", "No exercise-induced angina", "No")


# ----------------------------
# CP (chest pain)
# ----------------------------
def rule_cp(value):
    value = str(value).lower().strip()

    mapping = {
        "asymptomatic": ("severe", "Asymptomatic (silent ischemia, higher risk)"),
        "typical angina": ("moderate", "Typical angina"),
        "atypical angina": ("mild", "Atypical angina"),
        "non-anginal": ("normal", "Non-anginal chest pain")
    }

    # handle encoded values too (optional but smart)
    numeric_map = {
        "0": "typical angina",
        "1": "atypical angina",
        "2": "non-anginal",
        "3": "asymptomatic"
    }

    if value in numeric_map:
        value = numeric_map[value]

    sev, label = mapping.get(value, ("unknown", "Unknown chest pain type"))

    return build_output("cp", value, sev, label, "Categorical")


# ----------------------------
# THAL
# ----------------------------
def rule_thal(value):
    if "revers" in str(value).lower():
        return build_output("thal", value, "severe", "Reversible perfusion defect", "Ischemia")
    elif "fixed" in str(value).lower():
        return build_output("thal", value, "moderate", "Fixed perfusion defect", "Scar tissue")
    else:
        return build_output("thal", value, "normal", "Normal perfusion", "Normal")
    
    

def rule_fbs(value):
    value = int(value)

    if value == 1:
        return build_output("fbs", value, "severe", "Fasting blood sugar >120 mg/dL (diabetic range)", ">120 mg/dL")
    else:
        return build_output("fbs", value, "normal", "Normal fasting blood sugar", "<120 mg/dL")

    
    
def rule_restecg(value):
    value_str = str(value).lower()

    if "st" in value_str:
        return build_output("restecg", value, "moderate", "ST-T abnormality", "Abnormal")
    elif "lv" in value_str:
        return build_output("restecg", value, "severe", "Left ventricular hypertrophy", "Abnormal")
    else:
        return build_output("restecg", value, "normal", "Normal ECG", "Normal")

# ----------------------------
# THALCH (max heart rate)
# ----------------------------

def rule_thalch(value, age):
    value = float(value)

    if age is None:
        return build_output("thalch", value, "unknown", "Missing age", "220-age")

    expected = 220 - age
    if expected <= 0:
        return build_output("thalch", value, "unknown", "Invalid age input", "220-age")
    pct = (value / expected) * 100

    if pct < 70:
        severity = "severe"
        label = "Chronotropic incompetence"
    elif pct < 85:
        severity = "moderate"
        label = "Reduced heart rate response"
    else:
        severity = "normal"
        label = "Normal heart rate response"

    out = build_output("thalch", value, severity, label, ">=85% expected")

    # ✅ add derived values (used in narratives)
    out["expected"] = expected
    out["achievement_pct"] = pct

    return out

    

def rule_sex(value):
    if str(value).lower() in ["male", "1"]:
        return build_output("sex", value, "moderate", "Male (higher cardiovascular risk)", "Epidemiological risk")
    else:
        return build_output("sex", value, "normal", "Female (lower baseline risk)", "Epidemiological risk")    


# ----------------------------
# MASTER RULE DISPATCHER
# ----------------------------
def apply_rules(patient_row):
    """
    Apply all clinical rules to a patient row (pandas Series or dict)
    """
    age = patient_row.get("age", None)

    return {
        "oldpeak": rule_oldpeak(patient_row.get("oldpeak", 0)),
        "chol": rule_chol(patient_row.get("chol", 0)),
        "trestbps": rule_trestbps(patient_row.get("trestbps", 0)),
        "age": rule_age(patient_row.get("age", 0)),
        "ca": rule_ca(patient_row.get("ca", 0)),
        "thalch": rule_thalch(patient_row.get("thalch", 0), age),
        "exang": rule_exang(patient_row.get("exang", 0)),
        "cp": rule_cp(patient_row.get("cp", "")),
        "thal": rule_thal(patient_row.get("thal", "")),
        "fbs": rule_fbs(patient_row.get("fbs", 0)),            # ✅ added
        "restecg": rule_restecg(patient_row.get("restecg", "")),
        "sex": rule_sex(patient_row.get("sex", ""))
    }