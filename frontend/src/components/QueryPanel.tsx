import { useState } from "react";
import { runQuery, type QueryResult } from "../api";

const EXAMPLES = [
  "How many running events in the last 2 hours?",
  "Count events by action today",
  "Average confidence per action",
  "Show events per hour over time",
];

export function QueryPanel() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      setResult(await runQuery(q));
    } catch (e) {
      setResult({ question: q, sql: "", provider: "error", valid: false, error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  const columns = result?.rows && result.rows.length ? Object.keys(result.rows[0]) : [];

  return (
    <div className="panel">
      <h2>Ask the data</h2>
      <p className="muted">Natural language → safe SQL over the pose event log.</p>
      <div className="query-row">
        <input
          value={question}
          placeholder="e.g. how many aggressive events today?"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit(question)}
        />
        <button onClick={() => submit(question)} disabled={loading}>
          {loading ? "…" : "Run"}
        </button>
      </div>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => { setQuestion(ex); submit(ex); }}>
            {ex}
          </button>
        ))}
      </div>

      {result && (
        <div className="result">
          <div className="sql-box">
            <span className="tag">{result.provider}</span>
            <code>{result.sql || "—"}</code>
          </div>
          {result.error && <div className="error">{result.error}</div>}
          {columns.length > 0 && (
            <table>
              <thead>
                <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {result.rows!.slice(0, 20).map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => <td key={c}>{String(row[c])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {result.valid && columns.length === 0 && !result.error && (
            <div className="muted">No rows. Seed the DB with `make seed`.</div>
          )}
        </div>
      )}
    </div>
  );
}
