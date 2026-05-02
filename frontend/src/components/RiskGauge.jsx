export default function RiskGauge({ probability, riskLabel, mode }) {
  const pct = Math.round(probability * 100);
  const r = 66, c = 2 * Math.PI * r;
  const off = c - probability * c;
  const col = pct >= 70 ? "var(--risk-high)" : pct >= 40 ? "var(--risk-moderate)" : "var(--risk-low)";

  return (
    <div className="gauge">
      <div className="gauge__svg">
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle cx="80" cy="80" r={r} fill="none" stroke="var(--bg-input)" strokeWidth="10" />
          <circle cx="80" cy="80" r={r} fill="none" stroke={col} strokeWidth="10"
            strokeLinecap="round" strokeDasharray={c} strokeDashoffset={off}
            style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)", filter: `drop-shadow(0 0 6px ${col})` }} />
        </svg>
        <div className="gauge__center">
          <div className="gauge__pct" style={{ color: col }}>{pct}%</div>
          <div className="gauge__lbl" style={{ color: col }}>{riskLabel}</div>
        </div>
      </div>
      <div className="gauge__mode">{mode === "hybrid" ? "🔗 Hybrid Fusion" : "📊 XGB Only"}</div>
    </div>
  );
}
