# src/clinical/recommendation.py
"""
Maps predicted risk + top clinical factors → actionable guidance.

Two modes:
  - doctor : clinical terminology, specific investigations
  - patient : plain English, personal action steps

Input : result dict from explanation.generate_explanation()
Output: structured recommendation dict + human-readable text
"""


# ----------------------------
# Feature-specific advice
# ----------------------------
FEATURE_ADVICE = {
    "doctor": {
        "oldpeak": "Consider stress imaging or coronary angiography to assess ST changes.",
        "chol":    "Initiate or review lipid-lowering therapy (statin consideration).",
        "trestbps":"Optimise antihypertensive regimen; target BP <130/80 mmHg (ACC/AHA 2017).",
        "thalch":  "Evaluate chronotropic response; consider formal exercise capacity testing.",
        "ca":      "Multi-vessel involvement — revascularisation evaluation may be warranted.",
        "exang":   "Demand-induced ischemia confirmed — further functional imaging advised.",
        "cp":      "Asymptomatic presentation warrants proactive cardiac workup.",
        "thal":    "Reversible perfusion defect — nuclear stress imaging or MRI recommended.",
        "fbs":     "Screen for diabetes; HbA1c and glucose tolerance assessment advised.",
        "restecg": "Serial ECG monitoring and cardiology review recommended.",
        "age":     "Age-appropriate preventive cardiology measures recommended.",
        "sex":     "Consider sex-specific cardiovascular risk factors in evaluation.",
        "slope":   "Flat/downsloping ST slope warrants further ischemia evaluation.",
    },
    "patient": {
        "oldpeak": "Tell your doctor about any chest tightness or breathlessness during activity.",
        "chol":    "Reduce saturated fats, increase fibre, and ask your doctor about cholesterol medication.",
        "trestbps":"Reduce salt intake, exercise regularly, and monitor your blood pressure at home.",
        "thalch":  "Discuss your exercise capacity with your doctor — a cardiac fitness test may help.",
        "ca":      "Ask your doctor about the blocked arteries and what treatment options are available.",
        "exang":   "Avoid strenuous activity until you've spoken to your doctor about your symptoms.",
        "cp":      "Bring any history of chest discomfort to your doctor's attention, even if mild.",
        "thal":    "Ask your doctor to explain your heart scan results and what they mean for you.",
        "fbs":     "Check your blood sugar regularly and consider dietary changes to manage it.",
        "restecg": "Ask your doctor about your ECG result and whether follow-up testing is needed.",
        "age":     "Stay active, eat well, and attend regular heart health check-ups.",
        "sex":     "Discuss your personal cardiovascular risk profile with your doctor.",
        "slope":   "Ask your doctor to explain your stress test pattern and next steps.",
    }
}


# ----------------------------
# Core function
# ----------------------------
def get_recommendation(result, mode="patient"):
    """
    Generate recommendation from explanation result.

    Parameters:
        result : dict — output of generate_explanation()
        mode   : "doctor" or "patient"

    Returns:
        dict with keys: risk_level, urgency, action, detail,
                        key_focus_areas, text
    """
    risk_label   = result["risk_label"]
    risk_factors = result.get("risk_factors", [])
    top_features = [f["feature"] for f in risk_factors[:3]]
    if not top_features:
        feature_advice = []

    # ----------------------------
    # HIGH RISK
    # ----------------------------
    if risk_label == "HIGH RISK":
        urgency = "urgent"
        if mode == "doctor":
            action = "Urgent cardiology referral advised."
            detail = (
                "The combination of clinical indicators suggests significant coronary "
                "artery disease. Recommend cardiology consultation with consideration "
                "of further investigations: stress imaging, coronary angiography, "
                "and/or ECG monitoring. Review and optimise current medications."
            )
        else:
            action = "Please see a heart specialist (cardiologist) as soon as possible."
            detail = (
                "Your results show several concerning signs for your heart health. "
                "Do not ignore symptoms such as chest discomfort, shortness of breath, "
                "or unusual fatigue. Bring this report to your appointment. "
                "If symptoms worsen suddenly, seek emergency care immediately."
            )

    # ----------------------------
    # MODERATE RISK
    # ----------------------------
    elif risk_label == "MODERATE RISK":
        urgency = "medium"
        if mode == "doctor":
            action = "Cardiology or GP consultation recommended within 4 weeks."
            detail = (
                "Moderate risk profile with identifiable contributing factors. "
                "Consider formal cardiovascular risk scoring (Framingham/SCORE2), "
                "lifestyle intervention counselling, and targeted investigation "
                "of flagged risk factors."
            )
        else:
            action = "Book an appointment with your doctor to discuss your heart health."
            detail = (
                "Your results show some factors that may increase your heart risk. "
                "The good news is that many of these can be managed with lifestyle "
                "changes and medical support. Regular check-ups and monitoring "
                "are important steps forward."
            )

    # ----------------------------
    # LOW RISK
    # ----------------------------
    else:
        urgency = "low"
        if mode == "doctor":
            action = "Routine preventive monitoring. No immediate intervention required."
            detail = (
                "Current findings suggest lower cardiovascular risk. "
                "Continue standard preventive care: annual BP and lipid checks, "
                "lifestyle advice, and age-appropriate screening. "
                "Reassess risk in 1–2 years or if symptoms develop."
            )
        else:
            action = "Keep up your healthy habits — your heart looks good."
            detail = (
                "Your results suggest a lower risk of heart disease right now. "
                "The best thing you can do is maintain a balanced diet, "
                "stay physically active, avoid smoking, and attend your "
                "regular health check-ups to keep it that way."
            )

    # ----------------------------
    # Feature-specific advice
    # ----------------------------
    advice_pool = FEATURE_ADVICE[mode]
    feature_advice = [
        advice_pool[f]
        for f in top_features
        if f in advice_pool
    ]

    # ----------------------------
    # Build structured output
    # ----------------------------
    recommendation = {
        "risk_level":      risk_label,
        "urgency":         urgency,
        "action":          action,
        "detail":          detail,
        "key_focus_areas": feature_advice,
        "mode":            mode
    }

    # ----------------------------
    # Human-readable text
    # ----------------------------
    lines = []
    lines.append(f"RECOMMENDATION  [{urgency.upper()} PRIORITY]")
    lines.append("-" * 50)
    lines.append(f"{action}")
    lines.append("")
    lines.append(detail)

    if feature_advice:
        lines.append("")
        lines.append("Key areas to address:")
        for item in feature_advice:
            lines.append(f"  • {item}")
    
    lines.append("\n ⚕ This is a decision-support recommendation and does not replace clinical diagnosis.")

    recommendation["text"] = "\n".join(lines)

    return recommendation