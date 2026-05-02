# src/clinical/grouping.py

"""
Group SHAP values from one-hot encoded features
back to original clinical features safely.

Handles:
- One-hot encoded categorical features
- Numeric features
- Prefix collision issues (e.g., 'thal' vs 'thalch')
"""

# 🔥 IMPORTANT: longer names FIRST
BASE_FEATURES = sorted([
    "thalch", "restecg", "trestbps", "oldpeak",
    "thal", "slope", "cp", "sex", "chol",
    "age", "ca", "exang", "fbs"
], key=len, reverse=True)


def get_base_feature(feature_name):
    """
    Extract base feature name from transformed feature name.
    """

    for base in BASE_FEATURES:
        if feature_name == base or feature_name.startswith(base + "_"):
            return base

    # optional debug
    # print(f"[WARNING] Unmapped feature: {feature_name}")
    return feature_name


def group_shap_values(shap_values, feature_names):
    """
    Aggregate SHAP values back to base clinical features.

    Parameters:
        shap_values (array): SHAP values for one sample
        feature_names (list): transformed feature names

    Returns:
        dict: {base_feature: aggregated_shap_value}
    """

    grouped = {}

    for fname, val in zip(feature_names, shap_values):

        base = get_base_feature(fname)

        if base not in grouped:
            grouped[base] = 0.0

        grouped[base] += float(val)

    return grouped