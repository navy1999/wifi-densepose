"""Async Redis access: prediction cache + CSI frame stream.

Works against local Redis (docker) and serverless Upstash (rediss://) without
code changes. Cache failures are non-fatal — a cache miss must never take down
inference, so all cache ops swallow connection errors and degrade gracefully.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from wifipose.config import get_settings
from wifipose.logging_config import get_logger

log = get_logger("cache")

_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Lazy singleton client (connection pool under the hood)."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
    return _pool


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def csi_fingerprint(csi: np.ndarray) -> str:
    """Stable hash of a CSI window for cache keys (rounded to dedupe near-dupes)."""
    q = np.round(np.asarray(csi, dtype=np.float32), 2)
    return "pred:" + hashlib.sha1(q.tobytes()).hexdigest()[:16]


async def get_cached_prediction(key: str) -> dict[str, Any] | None:
    try:
        raw = await get_redis().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:  # noqa: BLE001 - cache must never be fatal
        log.warning("cache_get_failed", error=str(e))
        return None


async def set_cached_prediction(key: str, value: dict[str, Any], ttl: int) -> None:
    try:
        await get_redis().set(key, json.dumps(value), ex=ttl)
    except Exception as e:  # noqa: BLE001
        log.warning("cache_set_failed", error=str(e))
