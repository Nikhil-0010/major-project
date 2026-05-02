# backend/main.py
"""
FastAPI entry point for the Heart Disease CDSS backend.

Loads all model artifacts once at startup (lifespan event),
configures CORS for the frontend, and mounts the /predict route.
"""

import backend.confpath  # noqa: F401  — must be first to set sys.path

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.load_models import load_all_models
from backend.api.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    load_all_models()
    yield
    # Shutdown: nothing to clean up (models are in memory)


app = FastAPI(
    title="Heart Disease CDSS",
    description=(
        "Hybrid Clinical Decision Support System for Heart Disease Prediction. "
        "Uses XGBoost + ECG stream + meta-model fusion with SHAP-based explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Next.js fallback
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(predict_router, tags=["Prediction"])


@app.get("/health")
async def health_check():
    """Quick health check endpoint."""
    return {"status": "ok", "service": "heart-disease-cdss"}
