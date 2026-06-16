"""Shared dependencies (singletons) wired into routes via FastAPI Depends.

All are overridable with `app.dependency_overrides` in tests, which is how the
test suite runs the full API without a real model, database, or Redis.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException

from wifipose.config import get_settings
from wifipose.db.repository import PoseRepository
from wifipose.llm.base import LLMProvider
from wifipose.llm.providers import get_provider
from wifipose.logging_config import get_logger
from wifipose.ml.engine import PoseEstimator

log = get_logger("api")

_estimator: PoseEstimator | None = None
_estimator_error: str | None = None


def load_estimator() -> None:
    """Eager-load the ONNX model at startup (records error instead of crashing)."""
    global _estimator, _estimator_error
    settings = get_settings()
    try:
        _estimator = PoseEstimator(settings.model_path, intra_op_threads=settings.onnx_threads)
        log.info("estimator_loaded", path=settings.model_path)
    except Exception as e:  # noqa: BLE001
        _estimator_error = str(e)
        log.warning("estimator_unavailable", error=str(e))


def get_estimator() -> PoseEstimator:
    if _estimator is None:
        raise HTTPException(
            status_code=503,
            detail=f"Inference model unavailable: {_estimator_error or 'not loaded'}. "
            "Run `make checkpoint && make export`.",
        )
    return _estimator


def estimator_ready() -> bool:
    return _estimator is not None


@lru_cache
def get_repository() -> PoseRepository:
    return PoseRepository()


def get_llm() -> LLMProvider:
    return get_provider()
