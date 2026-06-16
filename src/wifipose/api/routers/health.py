"""Liveness / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from wifipose import __version__
from wifipose.api.deps import estimator_ready
from wifipose.config import get_settings
from wifipose.db.session import capabilities

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness — always 200 if the process is up."""
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready() -> dict:
    """Readiness — reports component availability for orchestrators/load balancers."""
    s = get_settings()
    caps = capabilities()
    return {
        "status": "ready",
        "model_loaded": estimator_ready(),
        "llm_provider": s.llm_provider,
        "db": {"timescaledb": caps.timescaledb, "pgvector": caps.pgvector},
    }
