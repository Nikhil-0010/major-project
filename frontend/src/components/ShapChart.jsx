import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine
} from "recharts";

export default function ShapChart({ features, values }) {
  if (!features?.length || !values?.length) return null;

  const data = features
    .map((n, i) => ({ name: n.replace(/_/g, " "), value: values[i] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <div className="shap-chart">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 90, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(15,23,42,0.06)" />
          <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "rgba(15,23,42,0.12)" }} />
          <YAxis type="category" dataKey="name" tick={{ fill: "#334155", fontSize: 10 }} axisLine={false} width={85} />
          <Tooltip
            contentStyle={{ background: "#ffffff", border: "1px solid rgba(15,23,42,0.1)", borderRadius: 8, color: "#0f172a", fontSize: 11 }}
            formatter={(v) => [v.toFixed(4), "SHAP"]}
          />
          <ReferenceLine x={0} stroke="rgba(15,23,42,0.15)" />
          <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={16}>
            {data.map((e, i) => (
              <Cell key={i} fill={e.value > 0 ? "#f87171" : "#34d399"} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
