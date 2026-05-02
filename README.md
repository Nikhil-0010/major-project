# Hybrid Clinical Decision Support System for Heart Disease Prediction

A full-stack web application that predicts heart disease risk using a hybrid AI system (XGBoost + ECG stream + meta-model), generates clinically grounded explanations using SHAP + rule-based interpretation, and presents results in both doctor-friendly and patient-friendly formats.

## How to Run Locally

### 1. Backend (FastAPI)
Open a terminal in the **root project directory**:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend (React + Vite)
Open a new terminal and navigate to the **frontend** directory:
```powershell
cd frontend
npm install
npm run dev
```

---

## Folder Structure

* `backend/` - FastAPI web server and prediction pipeline.
* `frontend/` - React frontend with Vite, Tailwind, and Recharts.
* `src/` - Core clinical logic, rules, and AI transformers.
* `artifacts/` - Pre-trained AI models (`.joblib` files) and Optuna configs.