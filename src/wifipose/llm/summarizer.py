"""Natural-language summaries of activity windows.

Given recent pose events (already fetched from the DB), produce a short
human-readable security/activity briefing. Uses the configured LLM when
available; falls back to a deterministic extractive summary offline.
"""

from __future__ import annotations

from collections import Counter

from wifipose.db.models import PoseEventOut
from wifipose.llm.base import LLMProvider

_SYSTEM = (
    "You are a security analytics assistant for a camera-free WiFi sensing system. "
    "Summarize the activity concisely (2-4 sentences), call out anything unusual "
    "(loitering, running, aggressive motion), and stay factual."
)


def _stats(events: list[PoseEventOut]) -> dict:
    counts = Counter(e.action for e in events)
    subjects = sorted({e.subject_id for e in events})
    avg_conf = sum(e.confidence for e in events) / len(events) if events else 0.0
    return {"counts": dict(counts), "subjects": subjects, "avg_confidence": round(avg_conf, 3)}


def _deterministic_summary(events: list[PoseEventOut]) -> str:
    if not events:
        return "No activity recorded in this window."
    s = _stats(events)
    counts = s["counts"]
    total = sum(counts.values())
    parts = [f"{n} {a.lower()}" for a, n in sorted(counts.items(), key=lambda x: -x[1])]
    summary = (
        f"{total} events across {len(s['subjects'])} subject(s): "
        + ", ".join(parts)
        + f". Mean confidence {s['avg_confidence']:.2f}."
    )
    alerts = []
    if counts.get("Aggressive"):
        alerts.append(f"{counts['Aggressive']} aggressive-motion event(s) detected")
    if counts.get("Loitering"):
        alerts.append(f"{counts['Loitering']} loitering event(s)")
    if alerts:
        summary += " Alert: " + "; ".join(alerts) + "."
    return summary


def _events_digest(events: list[PoseEventOut], cap: int = 40) -> str:
    lines = [
        f"- {e.ts.isoformat()} subject={e.subject_id} action={e.action} conf={e.confidence:.2f}"
        for e in events[:cap]
    ]
    return "\n".join(lines)


async def summarize_activity(events: list[PoseEventOut], provider: LLMProvider) -> dict:
    deterministic = _deterministic_summary(events)
    if provider.is_offline or not events:
        return {"summary": deterministic, "provider": "offline", "stats": _stats(events)}

    prompt = (
        "Summarize the following WiFi-sensing activity log.\n\n"
        f"{_events_digest(events)}\n\nBriefing:"
    )
    try:
        result = await provider.complete(prompt, system=_SYSTEM, temperature=0.2, max_tokens=220)
        return {"summary": result.text.strip(), "provider": result.provider, "stats": _stats(events)}
    except Exception:  # noqa: BLE001 - never let the LLM break the endpoint
        return {"summary": deterministic, "provider": "offline-fallback", "stats": _stats(events)}
