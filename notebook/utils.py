# 00_utils.py
import json, os, hashlib, joblib
from pathlib import Path
import numpy as np

SEED = 42

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_manifest(manifest_path, entry):
    mpath = Path(manifest_path)
    if mpath.exists():
        data = json.load(open(mpath))
    else:
        data = []
    data.append(entry)
    json.dump(data, open(mpath, "w"), indent=2)

def save_joblib(obj, path):
    joblib.dump(obj, path)
    return {"path": path, "sha256": sha256_file(path)}

def np_save(path, arr):
    np.save(path, arr)
    return {"path": path}
