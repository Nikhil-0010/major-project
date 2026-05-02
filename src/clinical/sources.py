"""
sources.py

Central registry of all medical references used in the system.
Every threshold and rule must map to one of these sources.

Purpose:
- Ensure clinical credibility
- Avoid hardcoding references across files
- Make project academically defensible
"""

SOURCES = {

    # Cardiology — Exercise testing / ECG interpretation
    "oldpeak": {
        "name": "ACC/AHA Exercise Testing Guidelines",
        "citation": "Gibbons RJ et al., Circulation, 2002",
        "url": "https://www.ahajournals.org/doi/10.1161/01.cir.0000034670.06526.15",
        "description": "Defines ST depression thresholds for myocardial ischemia during stress testing"
    },

    "exang": {
        "name": "ACC/AHA Exercise Testing Guidelines",
        "citation": "Gibbons RJ et al., Circulation, 2002",
        "url": "https://www.ahajournals.org/doi/10.1161/01.cir.0000034670.06526.15",
        "description": "Exercise-induced angina as indicator of ischemia"
    },

    "slope": {
        "name": "ACC/AHA Exercise Testing Guidelines",
        "citation": "Gibbons RJ et al., Circulation, 2002",
        "url": "https://www.ahajournals.org/doi/10.1161/01.cir.0000034670.06526.15",
        "description": "ST segment slope as predictor of ischemia severity"
    },

    # Chest pain classification
    "cp": {
        "name": "ACC/AHA Chest Pain Guidelines",
        "citation": "Gulati M et al., JACC, 2021",
        "url": "https://www.jacc.org/doi/10.1016/j.jacc.2021.07.053",
        "description": "Clinical classification of chest pain types and associated risk"
    },

    # Cholesterol
    "chol": {
        "name": "NCEP ATP III Guidelines",
        "citation": "National Cholesterol Education Program, 2001 (updated 2004)",
        "url": "https://www.nhlbi.nih.gov/health-topics/all-publications-and-resources/third-report-expert-panel-detection-evaluation-and-0",
        "description": "Defines cholesterol risk categories"
    },

    # Blood pressure
    "trestbps": {
        "name": "ACC/AHA Hypertension Guidelines",
        "citation": "Whelton PK et al., Hypertension, 2017",
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
        "description": "Defines blood pressure categories and cardiovascular risk"
    },

    # Heart rate response
    "thalch": {
        "name": "Chronotropic Incompetence Study",
        "citation": "Brubaker & Kitzman, Circulation, 2011",
        "url": "https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.110.985499",
        "description": "Defines abnormal heart rate response during exercise"
    },

    # Age risk
    "age": {
        "name": "Framingham Heart Study",
        "citation": "D'Agostino RB et al., Circulation, 2008",
        "url": "https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.107.699579",
        "description": "Age-based cardiovascular risk stratification"
    },

    # Blood sugar
    "fbs": {
        "name": "ADA Diabetes Guidelines",
        "citation": "American Diabetes Association, 2023",
        "url": "https://diabetesjournals.org/care/issue/46/Supplement_1",
        "description": "Defines fasting blood glucose thresholds"
    },

    # Coronary vessel blockage
    "ca": {
        "name": "AHA Coronary Artery Disease Guidelines",
        "citation": "Boden WE et al., COURAGE Trial + AHA 2021",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001038",
        "description": "Number of vessels involved and associated risk"
    },

    # Thalassemia / perfusion
    "thal": {
        "name": "ACC/AHA Nuclear Cardiology Guidelines",
        "citation": "Henzlova MJ et al., JACC, 2016",
        "url": "https://www.jacc.org/doi/10.1016/j.jacc.2016.09.002",
        "description": "Perfusion defects and myocardial ischemia"
    },

    # Resting ECG
    "restecg": {
        "name": "ACC/AHA ECG Interpretation Standards",
        "citation": "AHA/ACC ECG Standards",
        "url": "https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.108.191095",
        "description": "Resting ECG abnormalities and clinical relevance"
    }
}