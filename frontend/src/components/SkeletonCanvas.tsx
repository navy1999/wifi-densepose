import { useEffect, useRef } from "react";
import type { PoseFrame } from "../api";

// Body-part connection topology (matches the 14-joint limb map used in the
// research notebook's visualizations; extra parts are drawn as points).
const LIMBS: [number, number][] = [
  [0, 1],
  [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10],
  [1, 11], [11, 12], [12, 13],
];

const ACTION_COLORS: Record<string, string> = {
  Normal: "#4ade80",
  Loitering: "#facc15",
  Running: "#38bdf8",
  Aggressive: "#f87171",
};

export function SkeletonCanvas({ frame }: { frame: PoseFrame | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // grid backdrop
    ctx.strokeStyle = "rgba(148,163,184,0.12)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y <= H; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    if (!frame) {
      ctx.fillStyle = "#64748b";
      ctx.font = "16px system-ui";
      ctx.fillText("Waiting for live CSI stream…", 24, 32);
      return;
    }

    // UV coords are in [-1, 1]; map to canvas with padding.
    const pad = 60;
    const toXY = (u: number, v: number): [number, number] => [
      pad + ((u + 1) / 2) * (W - 2 * pad),
      pad + ((1 - (v + 1) / 2)) * (H - 2 * pad),
    ];

    const color = ACTION_COLORS[frame.action] ?? "#e2e8f0";

    // limbs
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    for (const [a, b] of LIMBS) {
      if (a >= frame.uv_coords.length || b >= frame.uv_coords.length) continue;
      const [ax, ay] = toXY(frame.uv_coords[a][0], frame.uv_coords[a][1]);
      const [bx, by] = toXY(frame.uv_coords[b][0], frame.uv_coords[b][1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }

    // joints, sized by confidence
    frame.uv_coords.forEach((pt, i) => {
      const [x, y] = toXY(pt[0], pt[1]);
      const conf = frame.confidence[i] ?? 0.5;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.35 + 0.65 * conf;
      ctx.beginPath();
      ctx.arc(x, y, 3 + 4 * conf, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }, [frame]);

  return <canvas ref={canvasRef} width={520} height={520} className="skeleton-canvas" />;
}
