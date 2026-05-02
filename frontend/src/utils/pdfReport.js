import jsPDF from "jspdf";

/**
 * Generate a professional clinical PDF report from prediction data.
 * Uses jsPDF's native API for clean text layout (no html2canvas dependency for reliability).
 */
export async function generatePDF(_, data, viewMode) {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const W = 210, M = 18, CW = W - 2 * M;
  let y = 0;

  const {
    probability, risk_label, mode, xgb_prob, ecg_prob,
    doctor_view, patient_view,
    risk_factors, protective_factors,
    shap_features, shap_values,
    recommendation_doctor, recommendation_patient,
    sources, model_info,
  } = data;

  const rec = viewMode === "doctor" ? recommendation_doctor : recommendation_patient;
  const narrative = viewMode === "doctor" ? doctor_view : patient_view;
  const pct = Math.round(probability * 100);
  const now = new Date().toLocaleString();

  // ── Helpers ──
  const rgb = (hex) => {
    const h = hex.replace("#", "");
    return [parseInt(h.substring(0, 2), 16), parseInt(h.substring(2, 4), 16), parseInt(h.substring(4, 6), 16)];
  };
  const riskCol = pct >= 70 ? "#dc2626" : pct >= 40 ? "#d97706" : "#059669";
  const riskBgCol = pct >= 70 ? "#fef2f2" : pct >= 40 ? "#fffbeb" : "#ecfdf5";

  const checkPage = (need = 20) => {
    if (y + need > 280) { doc.addPage(); y = 20; }
  };

  const drawLine = () => {
    doc.setDrawColor(200, 200, 200);
    doc.setLineWidth(0.3);
    doc.line(M, y, W - M, y);
    y += 4;
  };

  const sectionTitle = (text) => {
    checkPage(15);
    y += 3;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(30, 30, 30);
    doc.text(text, M, y);
    y += 2;
    drawLine();
  };

  const bodyText = (text, indent = 0) => {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(60, 60, 60);
    const paragraphs = text.split('\n');
    for (const para of paragraphs) {
      const trimmed = para.trimEnd();
      if (!trimmed) {
        y += 2; // small gap for empty lines
        continue;
      }
      const lines = doc.splitTextToSize(trimmed, CW - indent);
      for (const line of lines) {
        checkPage(5);
        doc.text(line, M + indent, y);
        y += 4.2;
      }
    }
  };

  // ══════════════════════════════════
  //  HEADER
  // ══════════════════════════════════
  // Blue header bar
  doc.setFillColor(37, 99, 235);
  doc.rect(0, 0, W, 28, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(255, 255, 255);
  doc.text("CardioInsight CDSS", M, 12);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.text("Clinical Decision Support System — Heart Disease Risk Assessment", M, 18);

  doc.setFontSize(7);
  doc.text(`Report generated: ${now}  |  View: ${viewMode === "doctor" ? "Clinician" : "Patient"}`, M, 24);

  y = 36;

  // ══════════════════════════════════
  //  RISK SUMMARY BOX
  // ══════════════════════════════════
  doc.setFillColor(...rgb(riskBgCol));
  doc.setDrawColor(...rgb(riskCol));
  doc.setLineWidth(0.5);
  doc.roundedRect(M, y, CW, 22, 3, 3, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(...rgb(riskCol));
  doc.text(`${pct}%`, M + 8, y + 14);

  doc.setFontSize(10);
  doc.text(risk_label, M + 30, y + 11);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(80, 80, 80);
  const modeStr = mode === "hybrid" ? "Hybrid Fusion (XGB + ECG)" : "XGBoost Only";
  doc.text(`Mode: ${modeStr}  |  XGB: ${(xgb_prob * 100).toFixed(1)}%${ecg_prob != null ? `  |  ECG: ${(ecg_prob * 100).toFixed(1)}%` : ""}  |  Final: ${(probability * 100).toFixed(1)}%`, M + 8, y + 19);

  y += 30;

  // ══════════════════════════════════
  //  MODEL PERFORMANCE
  // ══════════════════════════════════
  if (model_info) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(120, 120, 120);
    let perfText = "Model Performance: ";
    if (model_info.xgb_auc) perfText += `XGB AUC = ${model_info.xgb_auc.toFixed(3)}`;
    if (model_info.hybrid_auc) perfText += `  |  Hybrid AUC = ${model_info.hybrid_auc.toFixed(3)}`;
    doc.text(perfText, M, y);
    y += 6;
  }

  // ══════════════════════════════════
  //  RECOMMENDATION
  // ══════════════════════════════════
  if (rec) {
    sectionTitle("Clinical Recommendation");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8);
    doc.setTextColor(...rgb(riskCol));
    checkPage(6);
    doc.text(rec.urgency?.toUpperCase() || "", M, y);
    y += 5;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(9.5);
    doc.setTextColor(30, 30, 30);
    checkPage(6);
    doc.text(rec.action || "", M, y);
    y += 5;

    bodyText(rec.detail || "");

    if (rec.key_focus_areas?.length) {
      y += 2;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(60, 60, 60);
      checkPage(6);
      doc.text("Focus Areas: " + rec.key_focus_areas.join("  -  "), M, y);
      y += 6;
    }
  }

  // ══════════════════════════════════
  //  SHAP FEATURE CONTRIBUTIONS
  // ══════════════════════════════════
  if (shap_features?.length) {
    sectionTitle("Feature Contributions (SHAP)");

    const pairs = shap_features
      .map((n, i) => ({ name: n.replace(/_/g, " "), val: shap_values[i] }))
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val))
      .slice(0, 10);

    const maxAbs = Math.max(...pairs.map(p => Math.abs(p.val)), 0.01);
    const barW = 70;

    for (const p of pairs) {
      checkPage(7);
      // Feature name
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(60, 60, 60);
      doc.text(p.name, M, y + 3);

      // Bar
      const bLen = (Math.abs(p.val) / maxAbs) * barW;
      const bCol = p.val > 0 ? rgb("#ef4444") : rgb("#10b981");
      doc.setFillColor(...bCol);

      if (p.val >= 0) {
        doc.roundedRect(M + 55, y - 1, bLen, 5, 1, 1, "F");
      } else {
        doc.roundedRect(M + 55 + barW - bLen, y - 1, bLen, 5, 1, 1, "F");
      }

      // Value label
      doc.setFontSize(7);
      doc.setTextColor(100, 100, 100);
      doc.text(p.val.toFixed(4), M + 55 + barW + 3, y + 3);
      y += 7;
    }
    y += 2;
  }

  // ══════════════════════════════════
  //  RISK FACTORS
  // ══════════════════════════════════
  if (risk_factors?.length) {
    sectionTitle("Risk Factors");
    for (const f of risk_factors) {
      checkPage(12);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8.5);
      doc.setTextColor(180, 30, 30);
      doc.text("(+)", M, y);
      doc.setTextColor(40, 40, 40);
      const label = `${f.label || f.feature}${f.value ? ` (${f.value})` : ""}`;
      doc.text(label, M + 5, y);
      if (f.shap_impact) {
        doc.setFontSize(7);
        doc.setTextColor(150, 150, 150);
        doc.text(`SHAP: ${f.shap_impact > 0 ? "+" : ""}${f.shap_impact.toFixed(3)}`, M + CW - 25, y);
      }
      y += 4.5;
      const desc = viewMode === "doctor" ? f.doctor_explanation : f.patient_explanation;
      if (desc) bodyText(desc, 5);
      y += 1.5;
    }
  }

  // ══════════════════════════════════
  //  PROTECTIVE FACTORS
  // ══════════════════════════════════
  if (protective_factors?.length) {
    sectionTitle("Protective Factors");
    for (const f of protective_factors) {
      checkPage(12);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8.5);
      doc.setTextColor(5, 150, 105);
      doc.text("(-)", M, y);
      doc.setTextColor(40, 40, 40);
      const label = `${f.label || f.feature}${f.value ? ` (${f.value})` : ""}`;
      doc.text(label, M + 5, y);
      if (f.shap_impact) {
        doc.setFontSize(7);
        doc.setTextColor(150, 150, 150);
        doc.text(`SHAP: ${f.shap_impact > 0 ? "+" : ""}${f.shap_impact.toFixed(3)}`, M + CW - 25, y);
      }
      y += 4.5;
      const desc = viewMode === "doctor" ? f.doctor_explanation : f.patient_explanation;
      if (desc) bodyText(desc, 5);
      y += 1.5;
    }
  }

  // ══════════════════════════════════
  //  NARRATIVE
  // ══════════════════════════════════
  if (narrative) {
    sectionTitle(viewMode === "doctor" ? "Clinical Narrative" : "Patient Summary");
    bodyText(narrative);
    y += 3;
  }

  // ══════════════════════════════════
  //  REFERENCES
  // ══════════════════════════════════
  if (sources?.length) {
    sectionTitle("Medical References");
    for (const s of sources) {
      checkPage(10);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(50, 50, 50);
      doc.text(s.name, M, y);
      y += 3.5;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(7);
      doc.setTextColor(120, 120, 120);
      const citLines = doc.splitTextToSize(s.citation, CW - 4);
      for (const cl of citLines) {
        checkPage(4);
        doc.text(cl, M + 2, y);
        y += 3.5;
      }
      y += 2;
    }
  }

  // ══════════════════════════════════
  //  FOOTER
  // ══════════════════════════════════
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(6.5);
    doc.setTextColor(160, 160, 160);
    doc.text(
      `CardioInsight CDSS — For clinical reference only. Not a substitute for professional medical judgment.`,
      M, 290
    );
    doc.text(`Page ${i} of ${pageCount}`, W - M - 18, 290);
  }

  // ── Save ──
  const filename = `CardioInsight_Report_${risk_label.replace(/\s/g, "_")}_${Date.now()}.pdf`;
  doc.save(filename);
}
