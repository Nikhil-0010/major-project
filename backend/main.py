# backend/main.py
"""
FastAPI entry point for the Heart Disease CDSS backend.

Loads all model artifacts once at startup (lifespan event),
configures CORS for the frontend, and mounts the /predict route.
"""

import backend.confpath  # noqa: F401  — must be first to set sys.path

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
        "https://heart-prediction-sigma.vercel.app", # Vercel production frontend
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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page for the API."""
    frontend_url = os.getenv("FRONTEND_URL", "https://heart-prediction-sigma.vercel.app")
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CardioInsight CDSS API</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f8fafc;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                color: #0f172a;
            }}
            .container {{
                background: white;
                padding: 3rem;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                text-align: center;
                max-width: 500px;
                border: 1px solid rgba(15, 23, 42, 0.1);
            }}
            h1 {{
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #4f46e5, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            p {{
                color: #64748b;
                margin-bottom: 2rem;
                line-height: 1.5;
            }}
            a.button {{
                background-color: #4f46e5;
                color: white;
                padding: 12px 28px;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
                transition: background-color 0.2s, transform 0.1s;
                display: inline-block;
            }}
            a.button:hover {{
                background-color: #4338ca;
                transform: translateY(-1px);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CardioInsight CDSS</h1>
            <p>The prediction API is online and actively serving the machine learning models. Please proceed to the frontend interface to run clinical assessments.</p>
            <a class="button" href="{frontend_url}">Go to Application</a>
        </div>
    </body>
    </html>
    """
