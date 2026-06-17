"""Live CSI ingestion — Phase 2.

Self-service device registration (per-device API keys) plus authenticated
ingestion of CSI windows over HTTP and WebSocket. Windowing happens at the edge
(the capture agent), so the server stays stateless: it normalizes whatever CSI
format the device sends into the model's [SEQ_LEN, 104] contract and drops it
onto the same Redis Stream the inference workers consume — tagged with the
device id for multi-tenant fan-out.

    register:   POST /api/v1/devices            -> { api_key (shown once), ... }
    ingest:     POST /api/v1/ingest/window      (X-API-Key)  one window
                WS   /api/v1/ingest/stream      (?api_key=)  continuous windows
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from wifipose import SEQ_LEN
from wifipose.api.deps import get_device_repository
from wifipose.api.metrics import INGEST_FRAMES
from wifipose.api.security import (
    authenticate_device,
    generate_api_key,
    new_device_id,
    require_device,
)
from wifipose.cache.redis_client import get_redis
from wifipose.config import get_settings
from wifipose.db.models import DeviceCredentials, DeviceRegisterRequest
from wifipose.ingest.adapter import SUPPORTED_FORMATS, CSIFormatError, build_window
from wifipose.logging_config import get_logger

router = APIRouter(prefix="/api/v1", tags=["ingest"])
log = get_logger("ingest")


class IngestWindow(BaseModel):
    packets: list[list[float]] = Field(
        ..., description="CSI packets; each a raw per-subcarrier array (any length)"
    )
    format: str | None = Field(None, description=f"one of {SUPPORTED_FORMATS}; defaults to device")
    subject_id: str = "person"
    frame: int = 0


async def _emit_window(redis, device: dict, win: IngestWindow) -> dict:
    fmt = win.format or device["csi_format"]
    window = build_window(win.packets, fmt=fmt, seq_len=SEQ_LEN)  # -> [30, 104]
    settings = get_settings()
    await redis.xadd(
        settings.stream_key,
        {
            "device_id": device["id"],
            "subject_id": win.subject_id,
            "csi": json.dumps(window.tolist()),
            "source": "device",
            "frame": str(win.frame),
        },
        maxlen=10000,
        approximate=True,
    )
    INGEST_FRAMES.labels(device=device["id"]).inc()
    return {"accepted": True, "device_id": device["id"], "frame": win.frame}


# ── device registration ───────────────────────────────────────────────────────
@router.post("/devices", response_model=DeviceCredentials, status_code=201)
async def register_device(
    req: DeviceRegisterRequest, repo=Depends(get_device_repository)
) -> DeviceCredentials:
    """Register a capture device. The returned api_key is shown ONLY here."""
    if req.csi_format not in SUPPORTED_FORMATS:
        req.csi_format = "complex_interleaved"
    device_id = new_device_id()
    api_key, key_hash = generate_api_key()
    row = await repo.create_device(
        device_id=device_id,
        name=req.name,
        owner=req.owner,
        csi_format=req.csi_format,
        api_key_hash=key_hash,
    )
    log.info("device_registered", device_id=device_id, name=req.name)
    return DeviceCredentials(api_key=api_key, **row)


@router.get("/devices/me")
async def whoami(device=Depends(require_device)) -> dict:
    """Verify an API key and echo the device profile (used by the agent)."""
    return {k: device[k] for k in ("id", "name", "owner", "csi_format")}


# ── ingestion ─────────────────────────────────────────────────────────────────
@router.post("/ingest/window")
async def ingest_window(win: IngestWindow, device=Depends(require_device)) -> dict:
    """Ingest a single CSI window (HTTP). Normalized + queued for inference."""
    try:
        return await _emit_window(get_redis(), device, win)
    except CSIFormatError as e:
        return {"accepted": False, "error": str(e)}


@router.websocket("/ingest/stream")
async def ingest_stream(ws: WebSocket) -> None:
    """Continuous CSI ingestion (WebSocket). Auth via ?api_key= or X-API-Key."""
    api_key = ws.query_params.get("api_key") or ws.headers.get("x-api-key")
    repo = get_device_repository()
    try:
        device = await authenticate_device(api_key, repo)
    except Exception:  # noqa: BLE001 - reject unauthenticated sockets cleanly
        await ws.close(code=4401)
        return

    await ws.accept()
    try:
        await repo.touch_last_seen(device["id"])
    except Exception:  # noqa: BLE001
        pass

    redis = get_redis()
    try:
        while True:
            payload = await ws.receive_json()
            try:
                win = IngestWindow(**payload)
                ack = await _emit_window(redis, device, win)
            except (CSIFormatError, ValueError) as e:
                ack = {"accepted": False, "error": str(e)}
            await ws.send_json(ack)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("ingest_ws_error", device_id=device["id"], error=str(e))
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
