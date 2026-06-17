"""Analytics endpoints over stored pose events (time-series + vector search)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from wifipose.api.deps import get_llm, get_repository
from wifipose.db.models import PoseEventOut
from wifipose.llm.summarizer import summarize_activity

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/events", response_model=list[PoseEventOut])
async def list_events(
    limit: int = Query(50, ge=1, le=500),
    action: str | None = None,
    subject_id: str | None = None,
    repo=Depends(get_repository),
):
    return await repo.recent_events(limit=limit, action=action, subject_id=subject_id)


@router.get("/stats/histogram")
async def action_histogram(hours: int = Query(24, ge=1, le=720), repo=Depends(get_repository)):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {"since": since.isoformat(), "counts": await repo.action_histogram(since=since)}


@router.get("/stats/timeseries")
async def action_timeseries(
    hours: int = Query(24, ge=1, le=720),
    bucket: str = Query("1 hour"),
    repo=Depends(get_repository),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return await repo.action_timeseries(bucket=bucket, since=since)


@router.get("/similar/{event_id}")
async def similar_to_event(event_id: int, k: int = Query(5, ge=1, le=20), repo=Depends(get_repository)):
    """Find poses most similar to a stored event (pgvector ANN or cosine fallback)."""
    events = await repo.recent_events(limit=500)
    target = next((e for e in events if e.id == event_id), None)
    if target is None:
        return {"error": f"event {event_id} not found in recent window"}
    from wifipose.db.models import PoseEventIn

    embedding = PoseEventIn.embedding_from_uv(target.uv_coords)
    return await repo.find_similar_poses(embedding, k=k)


@router.get("/summary")
async def activity_summary(
    hours: int = Query(1, ge=1, le=168),
    limit: int = Query(200, ge=1, le=500),
    repo=Depends(get_repository),
    llm=Depends(get_llm),
):
    """LLM (or deterministic) natural-language briefing of recent activity."""
    try:
        events = await repo.recent_events(limit=limit)
    except Exception as e:  # noqa: BLE001 - DB unreachable should not 500 the demo
        return {
            "summary": "Activity data is temporarily unavailable (database not reachable).",
            "provider": "unavailable",
            "stats": {},
            "error": str(e),
        }
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    windowed = [e for e in events if e.ts >= cutoff]
    return await summarize_activity(windowed, llm)
