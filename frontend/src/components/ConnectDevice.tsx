import { useState } from "react";
import { registerDevice, type DeviceCredentials } from "../api";

const FORMATS = [
  ["complex_interleaved", "ESP32 (esp-csi raw)"],
  ["amp_phase", "Amplitude + phase"],
  ["amplitude", "Amplitude only"],
  ["complex_pairs", "Complex [re, im] pairs"],
];

export function ConnectDevice() {
  const [name, setName] = useState("");
  const [format, setFormat] = useState("complex_interleaved");
  const [creds, setCreds] = useState<DeviceCredentials | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function register() {
    if (!name.trim()) return;
    setBusy(true);
    setError("");
    try {
      setCreds(await registerDevice(name, format));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const origin = window.location.origin;

  return (
    <div className="panel">
      <h2>Connect your CSI device</h2>
      <p className="muted">
        Register a capture device (ESP32 / Raspberry Pi&nbsp;Nexmon / Intel&nbsp;5300), then run the
        agent to stream live CSI. No hardware? Use the <code>synthetic</code> source.
      </p>

      {!creds && (
        <>
          <div className="query-row">
            <input
              value={name}
              placeholder="device name e.g. living-room-esp32"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && register()}
            />
            <button onClick={register} disabled={busy}>
              {busy ? "…" : "Register"}
            </button>
          </div>
          <select className="device-format" value={format} onChange={(e) => setFormat(e.target.value)}>
            {FORMATS.map(([v, label]) => (
              <option key={v} value={v}>{label}</option>
            ))}
          </select>
          {error && <div className="error">{error}</div>}
        </>
      )}

      {creds && (
        <div className="result">
          <div className="error">Save this API key now — it is shown only once.</div>
          <div className="sql-box"><span className="tag">device</span><code>{creds.id}</code></div>
          <div className="sql-box"><span className="tag">api key</span><code>{creds.api_key}</code></div>
          <p className="muted">Install the agent and start streaming:</p>
          <pre className="agent-cmd">{`pip install "wifipose[agent]"

# no hardware — synthesize CSI:
wifipose-agent stream --server ${origin} \\
  --api-key ${creds.api_key} --source synthetic

# real ESP32 running esp-csi over USB serial:
wifipose-agent stream --server ${origin} \\
  --api-key ${creds.api_key} --source esp32 --port COM3`}</pre>
          <button onClick={() => setCreds(null)}>Register another</button>
        </div>
      )}
    </div>
  );
}
