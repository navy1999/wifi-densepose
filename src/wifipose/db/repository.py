"""Data-access layer (raw parameterized SQL via SQLAlchemy text()).

Raw SQL is deliberate here: it shows the actual time-series queries (time_bucket
/ date_trunc, BRIN-friendly ranges, pgvector ANN) and keeps the LLM text-to-SQL
path honest — the same query surface the model is allowed to target.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from sqlalchemy import text

from wifipose.db.models import ActionBucket, PoseEventIn, PoseEventOut, SimilarPose
from wifipose.db.session import capabilities, get_sessionmaker


class PoseRepository:
    """All persistence operations for pose events."""

    def __init__(self, sessionmaker=None):
        self._sm = sessionmaker or get_sessionmaker()

    # ── writes ────────────────────────────────────────────────────────────
    async def insert_event(self, event: PoseEventIn) -> PoseEventOut:
        caps = capabilities()
        params: dict[str, Any] = {
            "subject_id": event.subject_id,
            "action_index": event.action_index,
            "action": event.action,
            "confidence": event.confidence,
            "action_probs": json.dumps(event.action_probs),
            "uv_coords": json.dumps(event.uv_coords),
            "embedding": event.embedding,
            "latency_ms": event.latency_ms,
            "source": event.source,
            "ts": event.ts,
        }
        ts_expr = "COALESCE(:ts, now())"
        cols = (
            "ts, subject_id, action_index, action, confidence, action_probs, "
            "uv_coords, embedding, latency_ms, source"
        )
        vals = (
            f"{ts_expr}, :subject_id, :action_index, :action, :confidence, "
            ":action_probs::jsonb, :uv_coords::jsonb, :embedding, :latency_ms, :source"
        )
        if caps.pgvector:
            cols += ", embedding_vec"
            vals += ", :embedding_vec::vector"
            params["embedding_vec"] = "[" + ",".join(str(x) for x in event.embedding) + "]"

        sql = text(
            f"INSERT INTO pose_events ({cols}) VALUES ({vals}) "
            "RETURNING id, ts, subject_id, action_index, action, confidence, "
            "action_probs, uv_coords, latency_ms, source"
        )
        async with self._sm() as session:
            row = (await session.execute(sql, params)).mappings().one()
            await session.commit()
        return _row_to_event(row)

    # ── reads ─────────────────────────────────────────────────────────────
    async def recent_events(
        self, limit: int = 50, action: str | None = None, subject_id: str | None = None
    ) -> list[PoseEventOut]:
        clauses, params = [], {"limit": limit}
        if action:
            clauses.append("action = :action")
            params["action"] = action
        if subject_id:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = text(
            "SELECT id, ts, subject_id, action_index, action, confidence, action_probs, "
            f"uv_coords, latency_ms, source FROM pose_events {where} ORDER BY ts DESC LIMIT :limit"
        )
        async with self._sm() as session:
            rows = (await session.execute(sql, params)).mappings().all()
        return [_row_to_event(r) for r in rows]

    async def action_histogram(self, since: datetime | None = None) -> dict[str, int]:
        params: dict[str, Any] = {}
        where = ""
        if since:
            where = "WHERE ts >= :since"
            params["since"] = since
        sql = text(f"SELECT action, COUNT(*) AS n FROM pose_events {where} GROUP BY action")
        async with self._sm() as session:
            rows = (await session.execute(sql, params)).mappings().all()
        return {r["action"]: int(r["n"]) for r in rows}

    async def action_timeseries(
        self, bucket: str = "1 hour", since: datetime | None = None
    ) -> list[ActionBucket]:
        """Counts per action per time-bucket. Uses Timescale time_bucket when
        available, else portable date_trunc."""
        caps = capabilities()
        params: dict[str, Any] = {}
        where = ""
        if since:
            where = "WHERE ts >= :since"
            params["since"] = since
        if caps.timescaledb:
            params["bucket"] = bucket
            bucket_expr = "time_bucket(CAST(:bucket AS interval), ts)"
        else:
            # map common intervals to a date_trunc unit
            unit = "hour" if "hour" in bucket else "minute" if "min" in bucket else "day"
            bucket_expr = f"date_trunc('{unit}', ts)"
        sql = text(
            f"SELECT {bucket_expr} AS bucket, action, COUNT(*) AS n "
            f"FROM pose_events {where} GROUP BY bucket, action ORDER BY bucket"
        )
        async with self._sm() as session:
            rows = (await session.execute(sql, params)).mappings().all()
        return [ActionBucket(bucket=r["bucket"], action=r["action"], count=int(r["n"])) for r in rows]

    async def find_similar_poses(self, embedding: list[float], k: int = 5) -> list[SimilarPose]:
        caps = capabilities()
        if caps.pgvector:
            return await self._find_similar_pgvector(embedding, k)
        return await self._find_similar_python(embedding, k)

    async def _find_similar_pgvector(self, embedding: list[float], k: int) -> list[SimilarPose]:
        qv = "[" + ",".join(str(x) for x in embedding) + "]"
        sql = text(
            "SELECT id, ts, subject_id, action_index, action, confidence, action_probs, "
            "uv_coords, latency_ms, source, "
            "1 - (embedding_vec <=> :qv::vector) AS similarity "
            "FROM pose_events WHERE embedding_vec IS NOT NULL "
            "ORDER BY embedding_vec <=> :qv::vector LIMIT :k"
        )
        async with self._sm() as session:
            rows = (await session.execute(sql, {"qv": qv, "k": k})).mappings().all()
        return [SimilarPose(**_row_to_event(r).model_dump(), similarity=float(r["similarity"])) for r in rows]

    async def _find_similar_python(self, embedding: list[float], k: int) -> list[SimilarPose]:
        sql = text(
            "SELECT id, ts, subject_id, action_index, action, confidence, action_probs, "
            "uv_coords, latency_ms, source, embedding FROM pose_events ORDER BY ts DESC LIMIT 500"
        )
        async with self._sm() as session:
            rows = (await session.execute(sql)).mappings().all()
        scored = []
        for r in rows:
            sim = _cosine(embedding, list(r["embedding"]))
            scored.append((sim, r))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            SimilarPose(**_row_to_event(r).model_dump(), similarity=round(sim, 4))
            for sim, r in scored[:k]
        ]

    async def run_safe_select(self, sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
        """Execute a pre-validated read-only SELECT inside a read-only transaction
        with a statement timeout. Validation is the caller's responsibility
        (see wifipose.llm.nl_to_sql.validate_select)."""
        async with self._sm() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(text("SET LOCAL statement_timeout = '5s'"))
            rows = (await session.execute(text(sql))).mappings().all()
        return [dict(r) for r in rows[:max_rows]]


# ── helpers ──────────────────────────────────────────────────────────────────
def _row_to_event(r) -> PoseEventOut:
    return PoseEventOut(
        id=int(r["id"]),
        ts=r["ts"],
        subject_id=r["subject_id"],
        action_index=int(r["action_index"]),
        action=r["action"],
        confidence=float(r["confidence"]),
        action_probs=_as_dict(r["action_probs"]),
        uv_coords=_as_list(r["uv_coords"]),
        latency_ms=float(r["latency_ms"]),
        source=r["source"],
    )


def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else json.loads(v)


def _as_list(v) -> list:
    return v if isinstance(v, list) else json.loads(v)


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n])) or 1e-9
    nb = math.sqrt(sum(x * x for x in b[:n])) or 1e-9
    return dot / (na * nb)
