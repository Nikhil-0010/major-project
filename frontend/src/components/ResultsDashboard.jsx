import { useState, useRef, useCallback } from "react";
import RiskGauge from "./RiskGauge";
import ShapChart from "./ShapChart";
import { generatePDF } from "../utils/pdfReport";

export default function ResultsDashboard({ data }) {
  const [view, setView] = useState("patient");
  const [pdfLoading, setPdfLoading] = useState(false);
  const reportRef = useRef(null);

  const {
    probability, risk_label, mode, xgb_prob, ecg_prob,
    doctor_view, patient_view,
    risk_factors, protective_factors,
    shap_features, shap_values,
    recommendation_doctor, recommendation_patient,
    sources, model_info,
  } = data;

  const rec = view === "doctor" ? recommendation_doctor : recommendation_patient;
  const narrative = view === "doctor" ? doctor_view : patient_view;

  const urgCol = rec?.urgency?.toLowerCase().includes("urgent") ? "var(--risk-high)"
    : rec?.urgency?.toLowerCase().includes("moderate") ? "var(--risk-moderate)" : "var(--risk-low)";

  const handlePDF = useCallback(async () => {
    setPdfLoading(true);
    try {
      await generatePDF(reportRef.current, data, view);
    } catch (e) { console.error("PDF generation failed:", e); }
    finally { setPdfLoading(false); }
  }, [data, view]);

  return (
    <div style={{ marginTop: "1rem" }} ref={reportRef} id="report-area">
      {/* Top bar: toggle + actions */}
      <div className="results-bar">
        <div className="view-toggle">
          <button className={`view-toggle__btn ${view === "doctor" ? "view-toggle__btn--active" : ""}`}
            onClick={() => setView("doctor")}>🩺 Doctor</button>
          <button className={`view-toggle__btn ${view === "patient" ? "view-toggle__btn--active" : ""}`}
            onClick={() => setView("patient")}>👤 Patient</button>
        </div>
        <button className="btn btn-ghost" onClick={handlePDF} disabled={pdfLoading}
          style={{ fontSize: "0.75rem", padding: "0.4rem 0.85rem" }}>
          {pdfLoading ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Generating...</>
            : <>📄 Download PDF</>}
        </button>
      </div>

      <div className="results-grid">
        {/* ── LEFT COLUMN ── */}
        <div className="results-left">
          {/* Risk Gauge */}
          <div className="card">
            <div className="card__head">
              <div className="card__ico" style={{ background: "rgba(248,113,113,0.1)", color: "#f87171" }}>⚠️</div>
              <div className="card__ttl">Risk Assessment</div>
            </div>
            <RiskGauge probability={probability} riskLabel={risk_label} mode={mode} />
            <div className="stat-bar">
              <div className="stat-bar__item">
                <div className="stat-bar__label">XGB</div>
                <div className="stat-bar__val">{(xgb_prob * 100).toFixed(1)}%</div>
              </div>
              {ecg_prob != null && (
                <div className="stat-bar__item">
                  <div className="stat-bar__label">ECG</div>
                  <div className="stat-bar__val">{(ecg_prob * 100).toFixed(1)}%</div>
                </div>
              )}
              <div className="stat-bar__item">
                <div className="stat-bar__label">Final</div>
                <div className="stat-bar__val">{(probability * 100).toFixed(1)}%</div>
              </div>
            </div>
            {model_info && (
              <div className="stat-bar" style={{ marginTop: "0.4rem" }}>
                {model_info.xgb_auc && <div className="stat-bar__item">
                  <div className="stat-bar__label">XGB AUC</div>
                  <div className="stat-bar__val">{model_info.xgb_auc.toFixed(3)}</div>
                </div>}
                {model_info.hybrid_auc && <div className="stat-bar__item">
                  <div className="stat-bar__label">Hybrid AUC</div>
                  <div className="stat-bar__val">{model_info.hybrid_auc.toFixed(3)}</div>
                </div>}
              </div>
            )}
          </div>

          {/* Recommendation */}
          {rec && (
            <div className="card">
              <div className="card__head">
                <div className="card__ico" style={{ background: "rgba(52,211,153,0.1)", color: "#34d399" }}>💡</div>
                <div className="card__ttl">Recommendation</div>
              </div>
              <div className="rec" style={{ borderLeftColor: urgCol }}>
                <div className="rec__urgency" style={{ color: urgCol }}>{rec.urgency}</div>
                <div className="rec__action">{rec.action}</div>
                <div className="rec__detail">{rec.detail}</div>
                {rec.key_focus_areas?.length > 0 && (
                  <div className="rec__tags">
                    {rec.key_focus_areas.map((a, i) => <span key={i} className="rec__tag">{a}</span>)}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Sources */}
          {sources?.length > 0 && (
            <div className="card">
              <div className="card__head">
                <div className="card__ico" style={{ background: "rgba(251,191,36,0.1)", color: "#fbbf24" }}>📚</div>
                <div className="card__ttl">References</div>
              </div>
              {sources.map((s, i) => (
                <div key={i} className="src">
                  {s.url ? (
                    <a href={s.url} target="_blank" rel="noopener noreferrer" className="src__name src__link" style={{ display: 'block', color: 'var(--accent)', textDecoration: 'none' }} onMouseOver={(e) => e.target.style.textDecoration = 'underline'} onMouseOut={(e) => e.target.style.textDecoration = 'none'}>
                      {s.name} ↗
                    </a>
                  ) : (
                    <div className="src__name">{s.name}</div>
                  )}
                  <div className="src__cite">{s.citation}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="results-right">
          {/* SHAP */}
          <div className="card">
            <div className="card__head">
              <div className="card__ico" style={{ background: "rgba(167,139,250,0.1)", color: "#a78bfa" }}>📊</div>
              <div>
                <div className="card__ttl">Feature Contributions</div>
                <div className="card__ttl-sub">SHAP analysis — red increases risk, green reduces</div>
              </div>
            </div>
            <ShapChart features={shap_features} values={shap_values} />
          </div>

          {/* Risk + Protective Factors */}
          <div className="card">
            <div className="card__head">
              <div className="card__ico" style={{ background: "rgba(248,113,113,0.1)", color: "#f87171" }}>🔬</div>
              <div>
                <div className="card__ttl">Clinical Factors</div>
                <div className="card__ttl-sub">
                  {risk_factors?.length || 0} risk · {protective_factors?.length || 0} protective
                </div>
              </div>
            </div>

            {risk_factors?.length > 0 && (
              <>
                <div className="sec-label">Risk Factors</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "1rem" }}>
                  {risk_factors.map((f, i) => <Factor key={`r${i}`} f={f} type="risk" view={view} />)}
                </div>
              </>
            )}
            {protective_factors?.length > 0 && (
              <>
                <div className="sec-label">Protective Factors</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {protective_factors.map((f, i) => <Factor key={`p${i}`} f={f} type="safe" view={view} />)}
                </div>
              </>
            )}
          </div>

          {/* Narrative */}
          {narrative && (
            <div className="card">
              <div className="card__head">
                <div className="card__ico" style={{ background: "rgba(34,211,238,0.1)", color: "#22d3ee" }}>📝</div>
                <div className="card__ttl">{view === "doctor" ? "Clinical Narrative" : "Patient Summary"}</div>
              </div>
              <div className="narrative">{narrative}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Factor({ f, type, view }) {
  const desc = view === "doctor" ? f.doctor_explanation : f.patient_explanation;
  const sCol = f.shap_impact > 0 ? "var(--red)" : "var(--green)";
  const sBg = f.shap_impact > 0 ? "rgba(248,113,113,0.1)" : "rgba(52,211,153,0.1)";

  return (
    <div className={`factor factor--${type}`}>
      <span className="factor__ico">{type === "risk" ? "⚠️" : "✅"}</span>
      <div className="factor__body">
        <div className="factor__name">
          {f.label || f.feature}
          {f.value && <span>({f.value})</span>}
        </div>
        {desc && <div className="factor__desc">{desc}</div>}
      </div>
      {f.shap_impact != null && f.shap_impact !== 0 && (
        <span className="factor__shap" style={{ background: sBg, color: sCol }}>
          {f.shap_impact > 0 ? "+" : ""}{f.shap_impact.toFixed(3)}
        </span>
      )}
    </div>
  );
}
