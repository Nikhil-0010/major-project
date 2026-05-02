# backend/models/load_models.py
"""
Singleton model registry — loads all artifacts once at startup.

Handles:
- Preprocessor (fitted ColumnTransformer)
- XGB stream model (top-7 features)
- ECG stream model (ECG-related features)
- Meta model (fusion of XGB + ECG)
- Standalone XGB k15 model (for comparison)
- Feature index maps (dynamically derived)
- SHAP TreeExplainer (pre-built on XGB)
"""

import os
import json
import joblib
import numpy as np
import shap

import backend.confpath as confpath

# Import src.transforms so that custom classes (ColumnSelector, etc.)
# are available in the module namespace when joblib unpickles models
# that were saved with those classes embedded.
import src.transforms  # noqa: F401
from src.transforms import ColumnSelector, select_cols_indices, select_cols_top7

# The xgb_k15_final.joblib was pickled from a notebook where these classes
# lived in __main__. We must inject them into __main__ so joblib can find them.
import sys
import __main__
__main__.ColumnSelector = ColumnSelector
__main__.select_cols_indices = select_cols_indices
__main__.select_cols_top7 = select_cols_top7


class ModelRegistry:
    """Holds all loaded model artifacts and derived metadata."""

    def __init__(self):
        self.preprocessor = None
        self.xgb_model = None
        self.ecg_model = None
        self.meta_model = None
        self.xgb_k15 = None
        self.explainer = None

        # Derived at load time
        self.transformed_feature_names = []
        self.xgb_selected_features = []
        self.xgb_indices = []
        self.ecg_indices = []
        self.full_feature_names = []

        # Performance metrics (from fusion_decision.json)
        self.xgb_auc = None
        self.hybrid_auc = None

    def is_loaded(self):
        return self.preprocessor is not None


# Module-level singleton
registry = ModelRegistry()


# ECG base feature names (pre-OHE)
ECG_BASE_FEATURES = ["oldpeak", "exang", "slope", "thal"]


def _derive_xgb_indices(transformed_names, top_features):
    """
    Derive XGB stream indices using the same logic as the training notebook
    (07_ecg_hybrid.ipynb): match each top feature by checking if the
    transformed name ends with it.

    This handles both exact matches (e.g., "chol") and OHE-expanded
    matches (e.g., "cp_asymptomatic" ends with "cp_asymptomatic").
    """
    indices = []
    for feat in top_features:
        matches = [i for i, name in enumerate(transformed_names) if name.endswith(feat)]
        indices.extend(matches)
    return sorted(set(indices))


def _derive_ecg_indices(transformed_names):
    """
    Derive ECG stream indices from transformed feature names.
    ECG features are: oldpeak, exang_*, slope_*, thal_*
    (all OHE expansions of the ECG base features)

    Uses the same logic as the training notebook: `any(base in name ...)`
    """
    indices = [
        i for i, name in enumerate(transformed_names)
        if any(base in name for base in ECG_BASE_FEATURES)
    ]
    return sorted(set(indices))


def load_all_models():
    """
    Load every artifact from disk and populate the singleton registry.
    Call this once at application startup.
    """
    global registry

    if registry.is_loaded():
        print("[INFO] Models already loaded, skipping.")
        return registry

    print("[INFO] Loading model artifacts...")

    # --- Preprocessor ---
    preproc_path = os.path.join(confpath.PREPROCESSOR_DIR, "preprocessor_fixed.joblib")
    registry.preprocessor = joblib.load(preproc_path)
    print(f"  [OK] Preprocessor loaded from {preproc_path}")

    # --- Derive transformed feature names ---
    # get_feature_names_out() returns prefixed names like "num__age", "cat__cp_asymptomatic"
    # but the XGB/ECG models were trained with unprefixed names from feature_index_map.json.
    # We strip the prefixes to match.
    try:
        raw_names = list(registry.preprocessor.get_feature_names_out())
        registry.transformed_feature_names = [
            name.split("__", 1)[1] if "__" in name else name
            for name in raw_names
        ]
    except AttributeError:
        # Fallback: load from debug file (already unprefixed)
        debug_path = os.path.join(confpath.ARTIFACTS_DIR, "feature_names_debug.json")
        with open(debug_path, "r") as f:
            registry.transformed_feature_names = json.load(f)
    print(f"  [OK] {len(registry.transformed_feature_names)} transformed features derived")

    # --- XGB Stream ---
    xgb_path = os.path.join(confpath.MODELS_DIR, "xgb_k15_final.joblib")
    registry.xgb_model = joblib.load(xgb_path)
    print(f"  [OK] XGB stream loaded from {xgb_path}")

    # --- ECG Stream ---
    ecg_path = os.path.join(confpath.MODELS_DIR, "ecg_stream.joblib")
    registry.ecg_model = joblib.load(ecg_path)
    print(f"  [OK] ECG stream loaded from {ecg_path}")

    # --- Meta Model ---
    meta_path = os.path.join(confpath.MODELS_DIR, "meta_model.joblib")
    registry.meta_model = joblib.load(meta_path)
    print(f"  [OK] Meta model loaded from {meta_path}")

    # --- XGB k15 (standalone comparison) ---
    xgb_k15_path = os.path.join(confpath.MODELS_DIR, "xgb_k15_final.joblib")
    registry.xgb_k15 = joblib.load(xgb_k15_path)
    print(f"  [OK] XGB k15 loaded from {xgb_k15_path}")

    # --- Load XGB top features from Optuna config ---
    optuna_path = os.path.join(confpath.ARTIFACTS_DIR, "tuning_optuna", "optuna_k15_best.json")
    with open(optuna_path, "r") as f:
        optuna_meta = json.load(f)
    xgb_top_features = optuna_meta["top_features"]
    print(f"  [OK] Loaded {len(xgb_top_features)} XGB top features from Optuna config")

    # --- Derive feature indices dynamically ---
    # Uses the exact same matching logic as the training notebook (07_ecg_hybrid.ipynb)
    registry.xgb_selected_features = xgb_top_features
    registry.xgb_indices = _derive_xgb_indices(
        registry.transformed_feature_names, xgb_top_features
    )
    registry.ecg_indices = _derive_ecg_indices(registry.transformed_feature_names)
    print(f"  [OK] XGB indices ({len(registry.xgb_indices)}): {registry.xgb_indices}")
    print(f"  [OK] ECG indices ({len(registry.ecg_indices)}): {registry.ecg_indices}")

    # --- Load full feature names (pre-OHE, for reference) ---
    feat_map_path = os.path.join(confpath.ARTIFACTS_DIR, "feature_index_map.json")
    with open(feat_map_path, "r") as f:
        feat_map = json.load(f)
    registry.full_feature_names = feat_map.get("full_feature_names", [])

    # --- Performance metrics ---
    fusion_path = os.path.join(confpath.MODELS_DIR, "fusion_decision.json")
    if os.path.exists(fusion_path):
        with open(fusion_path, "r") as f:
            fusion = json.load(f)
        registry.xgb_auc = fusion.get("xgb_auc")
        registry.hybrid_auc = fusion.get("fused_auc")
        print(f"  [OK] XGB AUC: {registry.xgb_auc:.4f}, Hybrid AUC: {registry.hybrid_auc:.4f}")

    # --- SHAP Explainer (TreeExplainer on XGB stream) ---
    registry.explainer = shap.TreeExplainer(registry.xgb_model)
    print("  [OK] SHAP TreeExplainer created")

    print("[INFO] All models loaded successfully.\n")
    return registry
