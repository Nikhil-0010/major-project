# backend/confpath.py
"""
Path configuration for the backend.
Ensures backend can import from project root (src/) and find artifacts/.
"""

import sys
import os

# Resolve project root (one level up from backend/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add project root to sys.path so `from src.clinical...` works
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Canonical paths
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROCESSOR_DIR = os.path.join(ARTIFACTS_DIR, "preprocessor")
