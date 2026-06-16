"""WebSocket live pose stream — powers the real-time skeleton viewer.

Two modes (query param `mode`):
  - sim  (default): the endpoint synthesizes CSI, runs inference inline, and
    streams skeletons. Self-contained — needs no producer/worker/Redis, so the
    public demo always works.
  - live: tails the Redis pub/sub channel that the stream worker publishes to
    (the full producer -> Redis stream -> worker -> pub/sub pipeline).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from wifipose.api.deps import estimator_ready, get_estimator
from wifipose.api.metrics import INFERENCE_LATENCY, INFERENCE_TOTAL, WS_CONNECTIONS
from wifipose.cache.redis_client import get_redis
from wifipose.config import get_settings
from wifipose.logging_config import get_logger
from wifipose.ml.simulator import CSISimulator

router = APIRouter(tags=["stream"])
log = get_logger("ws")
PREDICTIONS_CHANNEL = "predictions"


@router.websocket("/api/v1/ws")
async def pose_stream(ws: WebSocket, mode: str = "sim", rate_hz: float | None = None) -> None:
    await ws.accept()
    WS_CONNECTIONS.inc()
    settings = get_settings()
    try:
        if mode == "live":
            await _stream_from_redis(ws)
        else:
            await _stream_simulated(ws, rate_hz or settings.producer_rate_hz)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("ws_error", error=str(e))
        with _suppress():
            await ws.close()
    finally:
        WS_CONNECTIONS.dec()


async def _stream_simulated(ws: WebSocket, rate_hz: float) -> None:
    if not estimator_ready():
        await ws.send_json({"error": "model not loaded; run `make checkpoint && make export`"})
        await ws.close()
        return
    estimator = get_estimator()
    sim = CSISimulator()
    interval = 1.0 / max(rate_hz, 0.5)
    subjects = max(1, get_settings().producer_subjects)
    frame = 0
    while True:
        subject = f"subject-{frame % subjects}"
        sample = sim.sample()
        pred = estimator.predict(sample.csi)
        INFERENCE_LATENCY.observe(pred.latency_ms)
        INFERENCE_TOTAL.labels(source="ws", action=pred.action_name).inc()
        await ws.send_json(
            {
                "subject_id": subject,
                "uv_coords": pred.uv_coords,
                "confidence": pred.confidence,
                "action": pred.action_name,
                "action_probs": pred.action_probs,
                "ground_truth_action": sample.action_name,
                "latency_ms": pred.latency_ms,
                "frame": frame,
            }
        )
        frame += 1
        await asyncio.sleep(interval)


async def _stream_from_redis(ws: WebSocket) -> None:
    redis = get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(PREDICTIONS_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            data = message["data"]
            await ws.send_text(data if isinstance(data, str) else json.dumps(data))
    finally:
        await pubsub.unsubscribe(PREDICTIONS_CHANNEL)
        await pubsub.aclose()


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
