# src/clinical/explanation.py

import numpy as np

from src.clinical.rules import apply_rules
from src.clinical.narratives import generate_narrative
from src.clinical.grouping import group_shap_values

ALWAYS_RISK_FEATURES = {
    "restecg",
    "thal",
    "exang",
    "cp"
}

CLINICAL_TYPES = {
    "restecg": "ecg_abnormality",
    "thal":    "perfusion_defect",
    "exang":   "exercise_ischemia",
    "cp":      "chest_pain_pattern",

    "chol":    "metabolic",
    "trestbps":"hemodynamic",
    "age":     "demographic",
    "oldpeak": "ischemia_marker",
    "ca":      "vascular_obstruction",
    "thalch":  "functional_capacity"
}


# ----------------------------
# Risk label
# ----------------------------
def get_risk_label(prob):
    if prob >= 0.7:
        return "HIGH RISK"
    elif prob >= 0.4:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


def get_impact_label(shap_val):
    """Convert raw SHAP to human-readable impact strength."""
    abs_val = abs(shap_val)
    if abs_val >= 0.5:
        return "Strong"
    elif abs_val >= 0.2:
        return "Moderate"
    elif abs_val >= 0.05:
        return "Mild"
    return "Minimal"


# ----------------------------
# Core function
# ----------------------------
def generate_explanation(
    patient_row,
    shap_values,
    feature_names,
    prob,
    top_k=5
):
    """
    Generate full explanation for a single patient.

    Parameters:
        patient_row  : pd.Series or dict — raw patient features
        shap_values  : 1D array — SHAP values for this patient
        feature_names: list — transformed (post-OHE) feature names
        prob         : float — predicted probability
        top_k        : int — max features to show per direction

    Returns:
        dict with keys:
            probability, risk_label,
            risk_factors, protective_factors,
            doctor_view, patient_view, json
    """

    # 1. Apply clinical rules to raw patient values
    rules = apply_rules(patient_row)

    # 2. Aggregate SHAP values by base clinical feature (handles OHE)
    aggregated = group_shap_values(shap_values, feature_names)

    # 3. Sort by absolute SHAP impact
    agg_pairs = sorted(
        aggregated.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # 4. Split into risk-increasing and protective
    pos = [(f, v) for f, v in agg_pairs if v > 0][:top_k]
    neg = [(f, v) for f, v in agg_pairs if v < 0][:top_k]

    top_features = sorted(pos + neg, key=lambda x: abs(x[1]), reverse=True)

    # 5. Early exit if nothing to explain
    if not top_features:
        return {
            "probability": float(prob),
            "risk_label": get_risk_label(prob),
            "risk_factors": [],
            "protective_factors": [],
            "doctor_view": "No significant contributing factors identified.",
            "patient_view": "No major factors influencing your risk were identified.",
            "json": {}
        }

    # 6. Build structured explanation items
    risk_factors = []
    protective_factors = []

    for feature, shap_val in top_features:
        if feature not in rules:
            continue

        rule = rules[feature]

        doctor_text = generate_narrative(rule, mode="doctor")
        patient_text = generate_narrative(rule, mode="patient")

        item = {
            "feature": feature,
            "value": rule["value"],
            "severity": rule["severity"],
            "label": rule.get("label", ""),
            "threshold": rule.get("threshold", ""),
            "shap_impact": float(shap_val),
            "impact_strength": get_impact_label(shap_val),
            "doctor_explanation": doctor_text,
            "patient_explanation": patient_text,
            "source": rule.get("source", {}).get("citation", ""),
            "clinical_type": CLINICAL_TYPES.get(feature, "clinical_measurement")
        }

        # Severity-first logic (clinically correct)
        if rule["severity"] == "normal":
            # clinically normal = genuinely protective
            protective_factors.append(item)
        elif shap_val < 0 and feature in ALWAYS_RISK_FEATURES:
            # inherently risky feature with negative SHAP
            # = "less severe than average" not truly protective
            item["muted"] = True
            item["muted_note"] = "Lower than average impact for this risk factor"
            risk_factors.append(item)
        elif shap_val < 0:
            # other features with negative SHAP = protective
            protective_factors.append(item)
        else:
            # positive SHAP = risk factor
            risk_factors.append(item)

    # 7. Sort each list by absolute impact
    risk_factors = sorted(
        risk_factors, key=lambda x: abs(x["shap_impact"]), reverse=True
    )
    protective_factors = sorted(
        protective_factors, key=lambda x: abs(x["shap_impact"]), reverse=True
    )

    # 8. Build result
    result = {
        "probability": float(prob),
        "risk_label": get_risk_label(prob),
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }

    # 9. Build both views
    result["doctor_view"] = build_doctor_view(result)
    result["patient_view"] = build_patient_view(result)
    result["json"] = {
        "probability": float(prob),
        "risk_label": get_risk_label(prob),
        "risk_factors": [
            {
                "feature": f["feature"],
                "value": f["value"],
                "severity": f["severity"],
                "impact_strength": f["impact_strength"],
                "shap_impact": f["shap_impact"],
                "explanation": f["patient_explanation"],
                "source": f["source"]
            }
            for f in risk_factors
        ],
        "protective_factors": [
            {
                "feature": f["feature"],
                "value": f["value"],
                "severity": f["severity"],
                "impact_strength": f["impact_strength"],
                "shap_impact": f["shap_impact"],
                "explanation": f["patient_explanation"],
                "source": f["source"]
            }
            for f in protective_factors
        ]
    }

    return result


# ----------------------------
# Doctor view — technical, with numbers + thresholds + sources
# ----------------------------
def build_doctor_view(result):
    W = 62  # box width
    lines = []

    prob = result["probability"] * 100
    risk = result["risk_label"]

    lines.append("=" * W)
    lines.append("CLINICAL RISK ASSESSMENT REPORT")
    lines.append(f"Predicted Risk: {prob:.1f}%  |  Classification: {risk}")
    lines.append("=" * W)

    if result["risk_factors"]:
        lines.append("")
        lines.append("RISK FACTORS  (ranked by model impact)")
        lines.append("-" * W)
        for i, f in enumerate(result["risk_factors"], 1):
            
            lines.append(
                f"  {i}. {f['feature'].upper():<12} "
                f"Value: {f['value']}  |  "
                f"Severity: {f['severity'].upper()}"
            )
            
            if f.get("muted"):
                lines.append(f"  ◦ {f['doctor_explanation']}  [lower impact]")
            else:
                lines.append(f"  • {f['doctor_explanation']}")
            
            if f["source"]:
                lines.append(f"     Ref: {f['source']}")
            lines.append(f"     SHAP impact: {f['shap_impact']:+.3f}  "
                         f"({f['impact_strength']} influence)")
            lines.append("")

    if result["protective_factors"]:
        lines.append("PROTECTIVE FACTORS")
        lines.append("-" * W)
        for i, f in enumerate(result["protective_factors"], 1):
            lines.append(
                f"  {i}. {f['feature'].upper():<12} "
                f"Value: {f['value']}  |  "
                f"Severity: {f['severity'].upper()}"
            )
            lines.append(f"     {f['doctor_explanation']}")
            if f["source"]:
                lines.append(f"     Ref: {f['source']}")
            lines.append(f"     SHAP impact: {f['shap_impact']:+.3f}  "
                         f"({f['impact_strength']} influence)")
            lines.append("")

    lines.append("=" * W)
    lines.append("Decision support tool. Clinical judgment required.")
    lines.append("=" * W)

    return "\n".join(lines)


# ----------------------------
# Patient view — plain language, no jargon, no raw SHAP
# ----------------------------
def build_patient_view(result):
    W = 52
    lines = []

    prob = result["probability"] * 100
    risk = result["risk_label"]

    lines.append("=" * W)
    lines.append("YOUR HEART HEALTH SUMMARY")
    lines.append("=" * W)
    lines.append(f"  Risk Score     : {prob:.1f}%")
    lines.append(f"  Risk Level     : {risk}")
    lines.append("=" * W)

    # Overall message
    lines.append("")
    if risk == "HIGH RISK":
        lines.append(
            "  Several concerning signs were detected in\n"
            "  your results. Please seek medical attention\n"
            "  promptly and share this report with your doctor."
        )
    elif risk == "MODERATE RISK":
        lines.append(
            "  Some risk factors are present. Speaking with\n"
            "  your doctor and making lifestyle changes can\n"
            "  help manage and reduce your risk."
        )
    else:
        lines.append(
            "  Your results suggest lower heart disease risk.\n"
            "  Keep maintaining a healthy lifestyle and attend\n"
            "  regular check-ups."
        )

    # Risk factors
    if result["risk_factors"]:
        lines.append("")
        lines.append("-" * W)
        lines.append("  WHY IS YOUR RISK AT THIS LEVEL?")
        lines.append("-" * W)
        for f in result["risk_factors"]:
            if f.get("muted"):
                lines.append(f"\n  ◦ [Minor factor]")
            else:
                lines.append(f"\n  ▲ [{f['impact_strength']} factor]")

            # wrap explanation at ~48 chars
            words = f["patient_explanation"].split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 50:
                    lines.append(line)
                    line = "    " + word + " "
                else:
                    line += word + " "
            if line.strip():
                lines.append(line)

    # Protective factors
    if result["protective_factors"]:
        lines.append("")
        lines.append("-" * W)
        lines.append("  WHAT IS WORKING IN YOUR FAVOUR?")
        lines.append("-" * W)
        for f in result["protective_factors"]:
            lines.append(f"\n  ▼ [{f['impact_strength']} factor]")
            words = f["patient_explanation"].split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 50:
                    lines.append(line)
                    line = "    " + word + " "
                else:
                    line += word + " "
            if line.strip():
                lines.append(line)

    lines.append("")
    lines.append("=" * W)
    lines.append(
        "  ⚕ This is a decision support tool only.\n"
        "    Your doctor makes all final decisions."
    )
    lines.append("=" * W)

    return "\n".join(lines)