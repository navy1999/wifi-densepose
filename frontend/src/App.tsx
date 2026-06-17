import { useEffect, useRef, useState } from "react";
import { SkeletonCanvas } from "./components/SkeletonCanvas";
import { QueryPanel } from "./components/QueryPanel";
import { ConnectDevice } from "./components/ConnectDevice";
import { openPoseStream, getSummary, type PoseFrame } from "./api";

export default function App() {
  const [frame, setFrame] = useState<PoseFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const [latencies, setLatencies] = useState<number[]>([]);
  const [summary, setSummary] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = openPoseStream(
      (f) => {
        setFrame(f);
        setConnected(true);
        setLatencies((prev) => [...prev.slice(-49), f.latency_ms]);
      },
      () => setConnected(false)
    );
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  useEffect(() => {
    getSummary(6).then((s) => setSummary(s.summary)).catch(() => {});
  }, []);

  const avgLatency = latencies.length
    ? (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(1)
    : "—";

  return (
    <div className="app">
      <header>
        <div>
          <h1>WiFi Pose Intelligence</h1>
          <p className="muted">Camera-free human pose &amp; action sensing from WiFi CSI</p>
        </div>
        <div className={`status ${connected ? "online" : "offline"}`}>
          {connected ? "● live" : "○ connecting"}
        </div>
      </header>

      <div className="grid">
        <section className="panel">
          <h2>Live skeleton</h2>
          <SkeletonCanvas frame={frame} />
          <div className="metrics">
            <Metric label="Subject" value={frame?.subject_id ?? "—"} />
            <Metric label="Action" value={frame?.action ?? "—"} highlight />
            <Metric label="Inference" value={`${frame?.latency_ms?.toFixed(1) ?? "—"} ms`} />
            <Metric label="Avg latency" value={`${avgLatency} ms`} />
          </div>
          {frame && (
            <div className="probs">
              {Object.entries(frame.action_probs).map(([k, v]) => (
                <div key={k} className="prob">
                  <span>{k}</span>
                  <div className="bar"><div style={{ width: `${v * 100}%` }} /></div>
                  <span className="pct">{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </section>

        <div className="col">
          <QueryPanel />
          <ConnectDevice />
          <section className="panel">
            <h2>Activity briefing</h2>
            <p className="summary">{summary || "Run `make seed` to populate activity, then refresh."}</p>
          </section>
        </div>
      </div>

      <footer className="muted">
        ONNX Runtime inference · TimescaleDB/pgvector · Redis streams · LLM analytics ·{" "}
        <a href="/docs">API docs</a> · <a href="/metrics">metrics</a>
      </footer>
    </div>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`metric ${highlight ? "metric-hi" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
