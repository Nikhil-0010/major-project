# src/artifacts.py
"""
Artifact management helpers for the Hybrid Heart Attack project.

Provides:
 - ensure_artifact_dirs()               # create canonical artifact directories
 - sha256_file(path)                    # compute sha256 checksum for a file
 - record_artifact(path, name, notes)   # append metadata entry to manifest.json
 - save_joblib_with_manifest(obj, path, name, notes)
 - save_json_with_manifest(data, path, name, notes)
 - save_numpy_with_manifest(arr, path, name, notes)
 - save_figure_with_manifest(fig, path, name, notes)  # fig is matplotlib Figure
 - load_manifest() and list_manifest()
"""

import os
import json
import time
import hashlib
import joblib
from pathlib import Path
import numpy as np

# ----------------- Configuration -----------------
ART_ROOT = os.path.join("..", "artifacts")  # default relative path used in notebooks
DIRS = [
    "preprocessor",
    "feature_selection",
    "models/xgb",
    "models/lr",
    "models/mlp",
    "shap/figures",
    "manifests",
    "checksums"
]
MANIFEST_PATH = os.path.join(ART_ROOT, "manifests", "manifest.json")

# ----------------- Directory helpers -----------------
def ensure_artifact_dirs(art_root: str = ART_ROOT):
    """
    Create canonical artifact directory structure under art_root.
    Call this once at the start of a notebook/run.
    """
    for d in DIRS:
        full = os.path.join(art_root, d)
        os.makedirs(full, exist_ok=True)
    # ensure manifest dir
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    # ensure manifest file exists
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w") as f:
            json.dump([], f, indent=2)
    return ART_ROOT

# ----------------- Checksum helper -----------------
def sha256_file(path: str) -> str:
    """Return SHA256 hex digest of file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------- Manifest helpers -----------------
def _read_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            return json.load(open(MANIFEST_PATH, "r"))
        except Exception:
            # if corrupted, return empty list
            return []
    return []

def _write_manifest(entries):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)

def record_artifact(path: str, name: str = None, notes: str = None, extras: dict = None):
    """
    Append an artifact entry to the manifest. Returns the entry dict.
    Fields: name, path, sha256, created_at, notes, extras
    """
    path = str(path)
    entry = {
        "name": name or Path(path).name,
        "path": path,
        "sha256": sha256_file(path) if os.path.exists(path) else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes or "",
    }
    if extras:
        entry["extras"] = extras
    entries = _read_manifest()
    entries.append(entry)
    _write_manifest(entries)
    return entry

def load_manifest():
    """Return list of manifest entries."""
    return _read_manifest()

def list_manifest():
    """Pretty-print manifest entries (returns list)."""
    entries = _read_manifest()
    for e in entries:
        print(f"- {e['name']} : {e['path']} (sha256={e.get('sha256')})")
    return entries

# ----------------- Save helpers (joblib / json / numpy / matplotlib) -----------------
def save_joblib_with_manifest(obj, path: str, name: str = None, notes: str = None, extras: dict = None):
    """
    Save object with joblib.dump and record it to manifest.
    Returns manifest entry.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return record_artifact(path, name=name, notes=notes, extras=extras)

def save_json_with_manifest(data, path: str, name: str = None, notes: str = None, extras: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return record_artifact(path, name=name, notes=notes, extras=extras)

def save_numpy_with_manifest(arr: np.ndarray, path: str, name: str = None, notes: str = None, extras: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    return record_artifact(path + ".npy" if not str(path).endswith(".npy") else path, name=name, notes=notes, extras=extras)

def save_figure_with_manifest(fig, path: str, name: str = None, notes: str = None, dpi: int = 250, extras: dict = None):
    """
    Save a matplotlib Figure instance to disk and record in manifest.
    Usage: save_figure_with_manifest(fig, "../artifacts/shap/figures/shap_bar.png", name="shap_bar")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return record_artifact(path, name=name, notes=notes, extras=extras)

# ----------------- Convenience wrappers -----------------
def save_model_pipeline(obj, model_name: str, alg: str = "xgb", calibrated: bool = False, art_root: str = ART_ROOT, extras: dict = None):
    """
    Save a model pipeline with a standardized filename and record to manifest.
    Example:
      save_model_pipeline(pipeline_xgb, "top7_pipeline", alg="xgb", calibrated=False)
    """
    subdir = os.path.join(art_root, "models", alg)
    os.makedirs(subdir, exist_ok=True)
    suffix = "calibrated" if calibrated else "baseline"
    fname = f"{model_name}_{suffix}.joblib"
    path = os.path.join(subdir, fname)
    return save_joblib_with_manifest(obj, path, name=f"{alg}_{model_name}_{suffix}", notes=f"algorithm={alg}, calibrated={calibrated}", extras=extras)

# ----------------- Small utility -----------------
def artifact_exists(path: str) -> bool:
    return os.path.exists(path)

# ----------------- Module self-test -----------------
if __name__ == "__main__":
    # quick local test if you run python src/artifacts.py
    print("Ensuring artifact directories...")
    ensure_artifact_dirs()
    print("Manifest path:", MANIFEST_PATH)
    print("Current manifest entries:", len(load_manifest()))
