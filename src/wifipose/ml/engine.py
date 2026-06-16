"""ONNX Runtime inference engine — the production serving path.

This module intentionally does NOT import torch. Serving only needs
onnxruntime + numpy, which keeps the container image ~10x smaller and cold
starts fast (important on free-tier hosts that sleep idle apps).

`PoseEstimator.predict` accepts a single CSI window and returns a typed result;
`predict_batch` vectorizes for the offline benchmark and the worker.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wifipose import ACTION_NAMES, NUM_FEATURES, NUM_PARTS, SEQ_LEN


@dataclass
class PosePrediction:
    uv_coords: list[list[float]]  # [24][2]
    confidence: list[float]  # [24]
    action_index: int
    action_name: str
    action_probs: dict[str, float]
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class PoseEstimator:
    """Loads a WiFi-DensePose ONNX model and runs CSI -> pose/action inference."""

    def __init__(self, model_path: str | Path, intra_op_threads: int = 1):
        import onnxruntime as ort  # local import keeps module import cheap

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Run `make checkpoint && make export` (or wifipose-export) first."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ── validation ──────────────────────────────────────────────────────────
    @staticmethod
    def validate_window(csi: np.ndarray) -> np.ndarray:
        """Coerce a single window to [1, SEQ_LEN, NUM_FEATURES] float32."""
        arr = np.asarray(csi, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[-1] != NUM_FEATURES or arr.shape[-2] != SEQ_LEN:
            raise ValueError(
                f"Expected CSI window of shape ({SEQ_LEN}, {NUM_FEATURES}); "
                f"got {tuple(arr.shape[-2:])}."
            )
        return arr

    # ── inference ───────────────────────────────────────────────────────────
    def _run(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        outs = self.session.run(self.output_names, {self.input_name: batch})
        return dict(zip(self.output_names, outs, strict=True))

    def predict(self, csi: np.ndarray) -> PosePrediction:
        batch = self.validate_window(csi)
        t0 = time.perf_counter()
        out = self._run(batch)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = out["action_logits"][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        return PosePrediction(
            uv_coords=out["uv_coords"][0].tolist(),
            confidence=out["confidence"][0].tolist(),
            action_index=idx,
            action_name=ACTION_NAMES[idx],
            action_probs={name: float(p) for name, p in zip(ACTION_NAMES, probs, strict=True)},
            latency_ms=round(latency_ms, 3),
        )

    def predict_batch(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        """Raw batched inference for benchmarking/throughput paths."""
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 3:
            raise ValueError("Batch must be [N, SEQ_LEN, NUM_FEATURES].")
        return self._run(batch)

    @property
    def num_parts(self) -> int:
        return NUM_PARTS
