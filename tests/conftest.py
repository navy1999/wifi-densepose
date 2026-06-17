"""Shared fixtures.

The API tests run the full app with NO external services: the repository, LLM,
and (optionally) the estimator are replaced via FastAPI dependency overrides
with in-memory fakes. This keeps CI hermetic and fast.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from wifipose import NUM_FEATURES, SEQ_LEN
from wifipose.db.models import PoseEventIn, PoseEventOut, SimilarPose


class FakeRepository:
    """In-memory stand-in for PoseRepository."""

    def __init__(self):
        self.events: list[PoseEventOut] = []
        self._id = 0

    async def insert_event(self, event: PoseEventIn) -> PoseEventOut:
        self._id += 1
        out = PoseEventOut(
            id=self._id,
            ts=event.ts or datetime.now(timezone.utc),
            subject_id=event.subject_id,
            action_index=event.action_index,
            action=event.action,
            confidence=event.confidence,
            action_probs=event.action_probs,
            uv_coords=event.uv_coords,
            latency_ms=event.latency_ms,
            source=event.source,
        )
        self.events.append(out)
        return out

    async def recent_events(self, limit=50, action=None, subject_id=None, device_id=None):
        rows = list(reversed(self.events))
        if action:
            rows = [e for e in rows if e.action == action]
        if subject_id:
            rows = [e for e in rows if e.subject_id == subject_id]
        if device_id:
            rows = [e for e in rows if e.device_id == device_id]
        return rows[:limit]

    async def action_histogram(self, since=None):
        out: dict[str, int] = {}
        for e in self.events:
            out[e.action] = out.get(e.action, 0) + 1
        return out

    async def action_timeseries(self, bucket="1 hour", since=None):
        return []

    async def find_similar_poses(self, embedding, k=5):
        return [SimilarPose(**e.model_dump(), similarity=1.0) for e in self.events[:k]]

    async def run_safe_select(self, sql, max_rows=200):
        return [{"sql_echo": sql, "rows": len(self.events)}]


class FakeDeviceRepository:
    """In-memory stand-in for DeviceRepository."""

    def __init__(self):
        self.by_hash: dict[str, dict] = {}

    async def create_device(self, device_id, name, owner, csi_format, api_key_hash):
        row = {
            "id": device_id,
            "name": name,
            "owner": owner,
            "csi_format": csi_format,
            "created_at": datetime.now(timezone.utc),
            "last_seen": None,
        }
        self.by_hash[api_key_hash] = row
        return row

    async def get_by_key_hash(self, api_key_hash):
        return self.by_hash.get(api_key_hash)

    async def touch_last_seen(self, device_id):
        return None


class FakeRedis:
    """Minimal async Redis stub capturing XADD calls."""

    def __init__(self):
        self.added: list[tuple] = []

    async def xadd(self, stream, fields, **kwargs):
        self.added.append((stream, fields))
        return f"{len(self.added)}-0"


@pytest.fixture
def csi_window() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((SEQ_LEN, NUM_FEATURES)).astype(np.float32)


@pytest.fixture
def fake_repo() -> FakeRepository:
    return FakeRepository()
