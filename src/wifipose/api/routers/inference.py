"""REST inference endpoints: run the model on a CSI window or a simulated sample."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from wifipose import NUM_FEATURES, SEQ_LEN
from wifipose.api.deps import get_estimator, get_repository
from wifipose.api.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    EVENTS_PERSISTED,
    INFERENCE_LATENCY,
    INFERENCE_TOTAL,
)
from wifipose.cache.redis_client import (
    csi_fingerprint,
    get_cached_prediction,
    set_cached_prediction,
)
from wifipose.config import get_settings
from wifipose.db.models import PoseEventIn
from wifipose.ml.engine import PoseEstimator
from wifipose.ml.simulator import CSISimulator

router = APIRouter(prefix="/api/v1", tags=["inference"])
_sim = CSISimulator()


class PredictRequest(BaseModel):
    csi: list[list[float]] = Field(
        ..., description=f"CSI window of shape [{SEQ_LEN}, {NUM_FEATURES}]"
    )
    subject_id: str = "api"
    persist: bool = False


class SimulateRequest(BaseModel):
    action: int | None = Field(None, ge=0, le=3, description="0=Normal 1=Loitering 2=Running 3=Aggressive")
    subject_id: str = "sim"
    persist: bool = True


async def _predict_and_maybe_store(
    csi: np.ndarray,
    subject_id: str,
    source: str,
    persist: bool,
    estimator: PoseEstimator,
    repo,
) -> dict:
    settings = get_settings()
    key = csi_fingerprint(csi)
    cached = await get_cached_prediction(key)
    if cached is not None:
        CACHE_HITS.inc()
        pred = cached
    else:
        CACHE_MISSES.inc()
        pred = estimator.predict(csi).as_dict()
        await set_cached_prediction(key, pred, settings.prediction_cache_ttl)

    INFERENCE_LATENCY.observe(pred["latency_ms"])
    INFERENCE_TOTAL.labels(source=source, action=pred["action_name"]).inc()

    if persist:
        mean_conf = float(np.mean(pred["confidence"])) if pred["confidence"] else 0.0
        event = PoseEventIn(
            subject_id=subject_id,
            action_index=pred["action_index"],
            action=pred["action_name"],
            confidence=mean_conf,
            action_probs=pred["action_probs"],
            uv_coords=pred["uv_coords"],
            embedding=PoseEventIn.embedding_from_uv(pred["uv_coords"]),
            latency_ms=pred["latency_ms"],
            source=source,
        )
        try:
            stored = await repo.insert_event(event)
            EVENTS_PERSISTED.inc()
            pred["event_id"] = stored.id
        except Exception as e:  # noqa: BLE001 - persistence is best-effort for the demo
            pred["persist_error"] = str(e)
    return pred


@router.post("/predict")
async def predict(
    req: PredictRequest,
    estimator: PoseEstimator = Depends(get_estimator),
    repo=Depends(get_repository),
) -> dict:
    """Run inference on a caller-supplied CSI window."""
    arr = np.asarray(req.csi, dtype=np.float32)
    try:
        PoseEstimator.validate_window(arr)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return await _predict_and_maybe_store(arr, req.subject_id, "api", req.persist, estimator, repo)


@router.post("/simulate")
async def simulate(
    req: SimulateRequest,
    estimator: PoseEstimator = Depends(get_estimator),
    repo=Depends(get_repository),
) -> dict:
    """Generate a synthetic CSI window and run inference — the keyless demo path."""
    sample = _sim.sample(action=req.action)
    pred = await _predict_and_maybe_store(
        sample.csi, req.subject_id, "api", req.persist, estimator, repo
    )
    pred["ground_truth_action"] = sample.action_name
    return pred
