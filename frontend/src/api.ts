// Thin API client. Uses relative URLs so it works in dev (Vite proxy) and in
// production (served by the same FastAPI process).

export interface PoseFrame {
  subject_id: string;
  uv_coords: number[][];
  confidence: number[];
  action: string;
  action_probs: Record<string, number>;
  ground_truth_action?: string;
  latency_ms: number;
  frame: number;
}

export interface QueryResult {
  question: string;
  sql: string;
  provider: string;
  valid: boolean;
  error?: string;
  rows?: Record<string, unknown>[];
  row_count?: number;
}

export function openPoseStream(onFrame: (f: PoseFrame) => void, onClose?: () => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/v1/ws?mode=sim`);
  ws.onmessage = (ev) => {
    try {
      onFrame(JSON.parse(ev.data));
    } catch {
      /* ignore malformed frame */
    }
  };
  ws.onclose = () => onClose?.();
  return ws;
}

export async function runQuery(question: string): Promise<QueryResult> {
  const resp = await fetch("/api/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, execute: true }),
  });
  return resp.json();
}

export async function getSummary(hours = 1): Promise<{ summary: string; provider: string }> {
  const resp = await fetch(`/api/v1/summary?hours=${hours}`);
  return resp.json();
}

export async function getHistogram(hours = 24): Promise<{ counts: Record<string, number> }> {
  const resp = await fetch(`/api/v1/stats/histogram?hours=${hours}`);
  return resp.json();
}

export interface DeviceCredentials {
  id: string;
  name: string;
  csi_format: string;
  api_key: string;
}

export async function registerDevice(name: string, csi_format: string): Promise<DeviceCredentials> {
  const resp = await fetch("/api/v1/devices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, csi_format }),
  });
  if (!resp.ok) throw new Error(`Registration failed (${resp.status}). Is the database connected?`);
  return resp.json();
}
