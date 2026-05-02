import { useState } from "react";
import { predictHeartDisease } from "../api/predict";

const INITIAL = {
  age: "", sex: "male", cp: "asymptomatic", trestbps: "", chol: "",
  fbs: 0, restecg: "normal", thalch: "", exang: 1,
  oldpeak: "", slope: "flat", ca: "", thal: "normal",
};

const SAMPLE = {
  age: 63, sex: "male", cp: "asymptomatic", trestbps: 145, chol: 250,
  fbs: 0, restecg: "normal", thalch: 150, exang: 1,
  oldpeak: 2.3, slope: "flat", ca: 2, thal: "reversable defect",
};

const FIELDS = [
  { name: "age",      label: "Age",                  type: "number", min: 20, max: 100, ph: "63" },
  { name: "sex",      label: "Sex",                  type: "select", opts: [["male","Male"],["female","Female"]] },
  { name: "cp",       label: "Chest Pain Type",      type: "select", opts: [["typical angina","Typical Angina"],["atypical angina","Atypical Angina"],["non-anginal","Non-Anginal"],["asymptomatic","Asymptomatic"]] },
  { name: "trestbps", label: "Resting BP (mmHg)",    type: "number", min: 80, max: 250, ph: "145" },
  { name: "chol",     label: "Cholesterol (mg/dl)",   type: "number", min: 100, max: 600, ph: "250" },
  { name: "fbs",      label: "Fasting BS > 120",     type: "select", opts: [[0,"No"],[1,"Yes"]] },
  { name: "restecg",  label: "Resting ECG",          type: "select", opts: [["normal","Normal"],["st-t abnormality","ST-T Abnormality"],["lv hypertrophy","LV Hypertrophy"]] },
  { name: "thalch",   label: "Max Heart Rate",       type: "number", min: 60, max: 220, ph: "150" },
  { name: "exang",    label: "Exercise Angina",      type: "select", opts: [[0,"No"],[1,"Yes"]] },
  { name: "oldpeak",  label: "ST Depression",        type: "number", min: 0, max: 10, step: 0.1, ph: "2.3" },
  { name: "slope",    label: "ST Slope",             type: "select", opts: [["upsloping","Upsloping"],["flat","Flat"],["downsloping","Downsloping"]] },
  { name: "ca",       label: "Major Vessels (0-3)",  type: "number", min: 0, max: 3, ph: "0" },
  { name: "thal",     label: "Thalassemia",          type: "select", opts: [["normal","Normal"],["fixed defect","Fixed Defect"],["reversable defect","Reversable Defect"]] },
];

export default function PatientForm({ onResults, loading, setLoading }) {
  const [form, setForm] = useState(INITIAL);
  const [error, setError] = useState(null);

  const set = (e) => {
    const { name, value, type } = e.target;
    setForm(p => ({ ...p, [name]: type === "number" ? (value === "" ? "" : Number(value)) : value }));
  };

  const submit = async (e) => {
    e.preventDefault(); setError(null); setLoading(true);
    try {
      const payload = {};
      for (const f of FIELDS)
        payload[f.name] = f.type === "number" ? Number(form[f.name]) : form[f.name];
      const data = await predictHeartDisease(payload);
      onResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Check if the backend is running.");
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ marginBottom: "1.5rem" }}>
      <div className="card__head">
        <div className="card__ico" style={{ background: "rgba(99,102,241,0.12)", color: "#818cf8" }}>📋</div>
        <div style={{ flex: 1 }}>
          <div className="card__ttl">Patient Clinical Data</div>
          <div className="card__ttl-sub">Enter 13 clinical parameters for risk assessment</div>
        </div>
        <button type="button" className="btn btn-ghost" onClick={() => setForm(SAMPLE)}
          style={{ fontSize: "0.75rem", padding: "0.35rem 0.7rem" }}>
          Load Sample
        </button>
      </div>

      <form onSubmit={submit}>
        <div className="form-grid">
          {FIELDS.map(f => (
            <div className="form-group" key={f.name}>
              <label htmlFor={f.name}>{f.label}</label>
              {f.type === "select" ? (
                <select id={f.name} name={f.name} value={form[f.name]} onChange={set}>
                  {f.opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              ) : (
                <input id={f.name} name={f.name} type="number"
                  min={f.min} max={f.max} step={f.step || 1}
                  placeholder={f.ph} value={form[f.name]} onChange={set} required />
              )}
            </div>
          ))}
        </div>

        {error && <div className="error-box">{error}</div>}

        <div className="btn-row">
          <button type="button" className="btn btn-ghost"
            onClick={() => { setForm(INITIAL); setError(null); }}>
            Clear
          </button>
          <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
            {loading ? <><span className="spinner" /> Analyzing...</> : <>🔍 Predict Risk</>}
          </button>
        </div>
      </form>
    </div>
  );
}
