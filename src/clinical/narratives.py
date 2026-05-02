# src/clinical/narratives.py
"""
Generates clinically meaningful explanations.

Each feature has:
  - doctor mode: value + threshold + guideline reference
  - patient mode: plain English, no jargon, actual numbers explained

Sources are referenced from sources.py via rule_output["source"].
"""

from src.clinical.sources import SOURCES
import numpy as np

def safe_number(val, default=0):
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def generate_narrative(rule_output, mode="doctor"):
    """
    Generate explanation text for a single feature.

    Parameters:
        rule_output : dict — output from rules.py (contains value, severity, source, etc.)
        mode        : "doctor" or "patient"

    Returns:
        str — explanation text
    """
    feature  = rule_output["feature"]
    value    = rule_output["value"]
    severity = rule_output["severity"]
    source   = rule_output.get("source", {})
    citation = source.get("citation", "")
    
    val = safe_number(rule_output.get("value"))

    # =========================================================
    # DOCTOR VIEW — clinical precision, thresholds, references
    # =========================================================
    if mode == "doctor":

        if feature == "oldpeak":
            return (
                f"ST depression: {val} mm "
                f"[Threshold: ≥2.0 mm -> severe ischemia | ≥1.0 mm -> significant | "
                f"Ref: {citation}]"
            )

        elif feature == "chol":
            return (
                f"Total cholesterol: {val} mg/dL "
                f"[≥240 = high risk | 200–239 = borderline | <200 = desirable | "
                f"Ref: {citation}]"
            )

        elif feature == "trestbps":
            return (
                f"Resting BP: {val} mmHg "
                f"[≥140 = Stage 2 HTN | 130–139 = Stage 1 | 120–129 = elevated | "
                f"Ref: {citation}]"
            )

        elif feature == "ca":
            return (
                f"Coronary vessels involved: {int(val)} "
                f"[0 = none | 1 = single-vessel | ≥2 = multi-vessel CAD | "
                f"Ref: {citation}]"
            )

        elif feature == "thalch":
            expected = rule_output.get("expected")
            pct      = rule_output.get("achievement_pct")
            if expected and pct:
                return (
                    f"Max HR: {val:.0f} bpm ({pct:.1f}% of expected {expected:.0f} bpm) "
                    f"[<85% = chronotropic incompetence | <62% = severe | "
                    f"Ref: {citation}]"
                )
            return f"Max HR: {val:.0f} bpm [age-adjusted expected unavailable]"

        elif feature == "exang":
            return (
                f"Exercise-induced angina: {'Present' if val == 1 else 'Absent'} "
                f"[Presence indicates demand-induced ischemia | Ref: {citation}]"
            )

        elif feature == "cp":
            return (
                f"Chest pain type: {val} "
                f"[Asymptomatic = highest risk (silent ischemia) | "
                f"Ref: {citation}]"
            )

        elif feature == "thal":
            return (
                f"Thallium/perfusion result: {val} "
                f"[Reversible defect = active ischemia | Fixed = scar | "
                f"Ref: {citation}]"
            )

        elif feature == "age":
            return (
                f"Age: {val} years "
                f"[≥65 = advanced risk | 50–64 = elevated | "
                f"Ref: {citation}]"
            )


        elif feature == "fbs":
            return (
                f"Fasting blood sugar: {'> 120 mg/dL (elevated)' if val == 1 else '≤ 120 mg/dL (normal)'} "
                f"[>120 mg/dL may indicate diabetes | Ref: {citation}]"
            )

        elif feature == "restecg":
            return (
                f"Resting ECG: {val} "
                f"[ST-T abnormality or LV hypertrophy = clinically significant | "
                f"Ref: {citation}]"
            )

        elif feature == "sex":
            return (
                f"Sex: {val} "
                f"[Male sex associated with higher baseline CAD risk in this population | "
                f"Ref: Framingham Heart Study]"
            )

        elif feature == "slope":
            return (
                f"ST slope: {val} "
                f"[Flat/downsloping = higher ischemia risk | "
                f"Ref: {citation}]"
            )

        # fallback
        return f"{feature}: {val} [No specific clinical threshold available]"


    # =========================================================
    # PATIENT VIEW — plain English, actual numbers, no jargon
    # =========================================================
    else:

        if feature == "oldpeak":

            if severity == "severe":
                return (
                    f"Your stress test showed significantly reduced blood flow "
                    f"to your heart ({val} mm reading — above the 2.0 mm "
                    f"concern level). This means your heart muscle may not be "
                    f"getting enough oxygen when it needs to work harder."
                )
            elif severity == "moderate":
                return (
                    f"Your stress test showed some reduced blood flow "
                    f"({val} mm). This is above the 1.0 mm monitoring threshold "
                    f"and worth discussing with your doctor."
                )
            elif severity == "mild":
                return (
                    f"A mild stress test reading ({val} mm) was detected. "
                    f"This is a minor finding below the main concern level "
                    f"but worth keeping an eye on."
                )
            else:
                return (
                    "Your stress test blood flow result is normal — "
                    "a reassuring finding."
                )

        elif feature == "ca":
            val = int(val)
            if val == 0:
                return (
                    "No blockages were found in your heart's main arteries. "
                    "This is a strong protective sign — your heart has good blood supply."
                )
            elif val == 1:
                return (
                    "1 of your heart's main arteries shows narrowing. "
                    "This reduces blood flow to part of your heart "
                    "and needs medical attention."
                )
            else:
                return (
                    f"{val} of your heart's main arteries show significant narrowing. "
                    f"This substantially reduces your heart's blood supply "
                    f"and requires prompt cardiology review."
                )

        elif feature == "thalch":
            expected = rule_output.get("expected")
            pct      = rule_output.get("achievement_pct")
            if expected and pct:
                if severity == "severe":
                    return (
                        f"During exercise your heart reached only {val:.0f} beats "
                        f"per minute, but {expected:.0f} bpm was expected for your age. "
                        f"You achieved just {pct:.0f}% of the expected rate. "
                        f"This suggests your heart struggles to keep up during "
                        f"physical activity — an important warning sign."
                    )
                elif severity == "moderate":
                    return (
                        f"Your heart rate during exercise ({val:.0f} bpm) was lower "
                        f"than expected ({expected:.0f} bpm for your age — {pct:.0f}% achieved). "
                        f"This may suggest reduced heart fitness or capacity."
                    )
                else:
                    return (
                        f"Your heart rate during exercise ({val:.0f} bpm) was "
                        f"healthy for your age ({pct:.0f}% of expected {expected:.0f} bpm). "
                        f"This is a reassuring sign of good cardiac response."
                    )
            return "Your heart rate response during exercise was below the expected level."

        elif feature == "chol":

            if severity == "severe":
                return (
                    f"Your cholesterol level ({val} mg/dL) is high "
                    f"(above the 240 mg/dL concern level). "
                    f"High cholesterol causes fatty deposits to build up in your arteries, "
                    f"slowly narrowing them and reducing blood flow to your heart."
                )
            elif severity == "moderate":
                return (
                    f"Your cholesterol ({val} mg/dL) is borderline high "
                    f"(200–239 mg/dL range). "
                    f"Making dietary changes now can help prevent it from rising further."
                )
            else:
                return (
                    f"Your cholesterol ({val} mg/dL) is in a healthy range — "
                    f"a positive factor for your heart health."
                )

        elif feature == "trestbps":

            if severity == "severe":
                return (
                    f"Your resting blood pressure ({val} mmHg) is high "
                    f"(above the 140 mmHg Stage 2 level). "
                    f"High blood pressure puts constant extra strain on your heart "
                    f"and arteries, increasing heart disease risk over time."
                )
            elif severity == "moderate":
                return (
                    f"Your blood pressure ({val} mmHg) is in the Stage 1 "
                    f"hypertension range (130–139 mmHg). "
                    f"Reducing salt, exercising regularly, and medical review can help."
                )
            elif severity == "mild":
                return (
                    f"Your blood pressure ({val} mmHg) is slightly elevated "
                    f"(120–129 mmHg range). "
                    f"Lifestyle changes can prevent this from worsening."
                )
            else:
                return (
                        f"Your blood pressure ({val} mmHg) is within the normal range — "
                    f"a positive heart health indicator."
                )

        elif feature == "exang":

            if severity == "severe":
                return (
                    "You experienced chest discomfort during exercise. "
                    "This is an important warning sign — it means your heart "
                    "is not receiving enough blood when it needs to work harder."
                )
            else:
                return (
                    "You did not experience chest discomfort during exercise — "
                    "a reassuring sign that your heart copes well with physical activity."
                )

            
        elif feature == "cp":

            if severity == "severe":
                return (
                    "Your results showed no obvious chest pain symptoms, even under stress. "
                    "This is called silent ischemia and is actually more concerning because "
                    "it often causes people to delay seeking help, despite significant "
                    "cardiac stress occurring."
                )
            elif severity == "moderate":
                return (
                    "You reported classic chest pain (angina) during exertion. "
                    "While uncomfortable, this is a well-recognised cardiac symptom "
                    "that your doctor should evaluate further."
                )
            elif severity == "mild":
                return (
                    "You reported atypical chest discomfort, which may or may not "
                    "be cardiac in origin. Your doctor can help determine the cause."
                )
            else:
                return (
                    "Your chest pain pattern appears non-cardiac in nature — "
                    "a reassuring finding."
                )

        elif feature == "thal":

            if severity == "severe":
                return (
                    "Your heart scan showed areas that receive less blood during "
                    "physical stress but recover at rest (reversible defect). "
                    "This is an important sign of reduced coronary blood flow "
                    "and warrants further evaluation."
                )
            elif severity == "moderate":
                return (
                    "Your heart scan showed a fixed area of reduced blood flow, "
                    "which may indicate old scarring from a previous cardiac event."
                )
            else:
                return (
                    "Your heart scan showed normal blood flow throughout — "
                    "a positive and reassuring finding."
                )

        elif feature == "age":
            
            if severity == "severe":
                return (
                    f"At {val:.0f} years old, age is a contributing factor. "
                    f"Cardiovascular risk naturally increases with age as arteries "
                    f"become stiffer and plaque builds up over time. "
                    f"Regular monitoring becomes more important."
                )
            elif severity == "moderate":
                return (
                    f"At {val:.0f} years old, your age contributes moderately "
                    f"to cardiovascular risk. Staying active and attending "
                    f"regular check-ups is recommended."
                )
            else:
                return (
                    f"Your age ({val:.0f}) carries relatively low cardiovascular "
                    f"risk on its own — an advantage in your overall profile."
                )

        elif feature == "fbs":
            if severity != "normal":
                return (
                    "Your fasting blood sugar was elevated (above 120 mg/dL), "
                    "which may indicate diabetes or pre-diabetes. "
                    "High blood sugar gradually damages blood vessels, "
                    "increasing heart disease risk over time."
                )
            else:
                return (
                    "Your fasting blood sugar is within the normal range — "
                    "a healthy metabolic indicator."
                )

        elif feature == "restecg":
            if severity != "normal":
                return (
                    "Your resting heart electrical activity showed some abnormality. "
                    "While this alone does not confirm heart disease, it warrants "
                    "further evaluation by your doctor."
                )
            else:
                return (
                    "Your resting heart electrical activity (ECG) appears normal — "
                    "no baseline electrical abnormalities detected."
                )

        elif feature == "sex":
            return (
                "Your demographic profile is one of several factors considered "
                "in calculating your overall risk score."
            )

        elif feature == "slope":
            if severity in ("severe", "moderate"):
                return (
                    "The pattern of your heart's electrical response during stress "
                    "suggests a higher-risk profile, which your doctor should review."
                )
            else:
                return (
                    "Your heart's electrical response pattern during stress "
                    "appears in a lower-risk category."
                )

        # fallback
        return "This factor contributed to your overall risk assessment."


# ----------------------------
# Batch generation
# ----------------------------
def generate_all_narratives(rules_dict, mode="doctor"):
    """Generate explanations for all features in a rules dict."""
    return {
        feature: generate_narrative(rule, mode=mode)
        for feature, rule in rules_dict.items()
    }