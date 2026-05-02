import { useState, useRef } from "react";
import "./index.css";
import PatientForm from "./components/PatientForm";
import ResultsDashboard from "./components/ResultsDashboard";

export default function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const resultsRef = useRef(null);

  const handleResults = (data) => {
    setResults(data);
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  };

  return (
    <>
      <header className="header">
        <div className="header__inner">
          <div className="header__brand">
            <div className="header__logo">🫀</div>
            <div>
              <div className="header__name">CardioInsight CDSS</div>
              <div className="header__sub">Clinical Decision Support System</div>
            </div>
          </div>
          <div className="header__tag">AI-Powered</div>
        </div>
      </header>

      <main className="main">
        <div className="wrap">
          <PatientForm onResults={handleResults} loading={loading} setLoading={setLoading} />
          <div ref={resultsRef}>
            {results && <ResultsDashboard data={results} />}
          </div>
        </div>
      </main>
    </>
  );
}
