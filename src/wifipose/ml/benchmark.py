"""Latency/throughput benchmark: PyTorch eager vs ONNX Runtime (CPU).

Produces the headline number for the README — the optimization story. Reports
p50/p95/p99 single-frame latency and batched throughput for both backends, and
writes a JSON artifact that the README / CI can consume.

Usage:
    wifipose-benchmark --checkpoint checkpoints/wifipose.pth --onnx checkpoints/wifipose.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wifipose import NUM_FEATURES, SEQ_LEN


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    return {
        "mean_ms": round(float(a.mean()), 3),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
    }


def _bench_loop(run, x, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        run(x)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run(x)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def benchmark(
    checkpoint: str | Path | None,
    onnx_path: str | Path,
    iters: int = 200,
    warmup: int = 20,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
) -> dict:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    single = rng.standard_normal((1, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
    results: dict = {"single_frame": {}, "throughput": {}}

    # ── ONNX Runtime ──
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def onnx_run(x):
        return sess.run(None, {in_name: x})

    results["single_frame"]["onnxruntime"] = _percentiles(_bench_loop(onnx_run, single, iters, warmup))

    # ── PyTorch eager (optional; only if torch + checkpoint present) ──
    torch_run = None
    if checkpoint and Path(checkpoint).exists():
        try:
            import torch

            from wifipose.ml.model import build_model

            model = build_model()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            model.eval()
            torch.set_num_threads(1)
            single_t = torch.from_numpy(single)

            def torch_run(_x):  # noqa: ARG001
                with torch.no_grad():
                    return model(single_t)

            results["single_frame"]["pytorch"] = _percentiles(
                _bench_loop(lambda _x: torch_run(_x), single, iters, warmup)
            )
        except ImportError:
            results["single_frame"]["pytorch"] = "torch not installed"

    # ── speedup ──
    if "pytorch" in results["single_frame"] and isinstance(
        results["single_frame"]["pytorch"], dict
    ):
        pt = results["single_frame"]["pytorch"]["p50_ms"]
        ox = results["single_frame"]["onnxruntime"]["p50_ms"]
        results["speedup_p50"] = round(pt / ox, 2) if ox else None

    # ── throughput across batch sizes (ONNX) ──
    for bs in batch_sizes:
        x = rng.standard_normal((bs, SEQ_LEN, NUM_FEATURES)).astype(np.float32)
        lat = _bench_loop(onnx_run, x, max(iters // 4, 20), warmup)
        med = float(np.percentile(lat, 50))
        results["throughput"][f"batch_{bs}"] = {
            "p50_ms": round(med, 3),
            "frames_per_sec": round(bs / (med / 1000.0), 1),
        }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime.")
    p.add_argument("--checkpoint", default="checkpoints/wifipose.pth")
    p.add_argument("--onnx", default="checkpoints/wifipose.onnx")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default="checkpoints/benchmark.json")
    a = p.parse_args()

    res = benchmark(a.checkpoint, a.onnx, iters=a.iters)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
