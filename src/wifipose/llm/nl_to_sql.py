"""Natural-language -> SQL over the pose_events time-series.

Security model (text-to-SQL is the dangerous part, so this is the careful part):
  1. The LLM is given ONLY the pose_events schema and told to emit one SELECT.
  2. Whatever comes back is run through `validate_select`, a strict allowlist
     validator: single statement, SELECT/WITH only, no DDL/DML, no comments,
     no multi-statement, table must be pose_events, and a LIMIT is enforced.
  3. The query executes in a READ ONLY transaction with a statement timeout
     (see repository.run_safe_select). Defense in depth: even a validator bug
     cannot mutate data.

With the offline provider, a deterministic rule-based generator handles the
common analytics intents — so the feature works with zero API keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wifipose import ACTION_NAMES
from wifipose.llm.base import LLMProvider

ALLOWED_TABLE = "pose_events"
ALLOWED_COLUMNS = {
    "ts", "id", "subject_id", "action_index", "action",
    "confidence", "action_probs", "uv_coords", "latency_ms", "source",
}
FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "truncate", "create",
    "grant", "revoke", "copy", "merge", "call", "do", "vacuum", "analyze",
    "comment", "set", "reset", "begin", "commit", "rollback", "pg_sleep",
    "information_schema", "pg_catalog", "pg_read_file", "dblink",
}
DEFAULT_LIMIT = 200

SCHEMA_DESCRIPTION = f"""
Table: {ALLOWED_TABLE}  (one row per pose/action inference event)
Columns:
  ts            TIMESTAMPTZ  -- event time
  subject_id    TEXT         -- which tracked person
  action        TEXT         -- one of {list(ACTION_NAMES)}
  action_index  SMALLINT     -- 0..3 matching the action above
  confidence    REAL         -- mean body-part confidence (0..1)
  latency_ms    REAL         -- model inference latency
  source        TEXT         -- 'stream' | 'api' | 'seed'
Time helpers available: now(), interval '<n> hours/minutes/days', date_trunc('hour', ts).
""".strip()


@dataclass
class SQLGeneration:
    question: str
    sql: str
    provider: str
    valid: bool
    error: str | None = None


class UnsafeQueryError(ValueError):
    """Raised when generated SQL fails the safety allowlist."""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def validate_select(sql: str) -> str:
    """Return a normalized, LIMIT-enforced SELECT or raise UnsafeQueryError."""
    sql = _strip_code_fence(sql).strip().rstrip(";").strip()
    if not sql:
        raise UnsafeQueryError("Empty query.")

    lowered = sql.lower()
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise UnsafeQueryError("Comments are not allowed.")
    if ";" in sql:
        raise UnsafeQueryError("Multiple statements are not allowed.")
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeQueryError("Only SELECT/WITH queries are allowed.")

    # whole-word forbidden keyword check
    words = set(re.findall(r"[a-z_]+", lowered))
    bad = words & FORBIDDEN
    if bad:
        raise UnsafeQueryError(f"Disallowed keyword(s): {', '.join(sorted(bad))}")

    # the only real table that may appear after FROM/JOIN is pose_events
    for ref in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", lowered):
        if ref != ALLOWED_TABLE:
            raise UnsafeQueryError(f"Only the '{ALLOWED_TABLE}' table may be queried (saw '{ref}').")

    # enforce a LIMIT to bound result size
    if not re.search(r"\blimit\s+\d+", lowered):
        sql = f"{sql} LIMIT {DEFAULT_LIMIT}"
    return sql


# ── offline deterministic generator ──────────────────────────────────────────
def rule_based_sql(question: str) -> str:
    """Pattern-match common analytics intents without an LLM."""
    q = question.lower()
    window = _extract_window(q)
    where_time = f"ts >= now() - interval '{window}'" if window else None

    # action filter mentioned?
    action = next((a for a in ACTION_NAMES if a.lower() in q), None)

    # "how many / count"
    if any(k in q for k in ("how many", "count", "number of", "times")):
        clauses = []
        if action:
            clauses.append(f"action = '{action}'")
        if where_time:
            clauses.append(where_time)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        if action:
            return f"SELECT COUNT(*) AS count FROM {ALLOWED_TABLE}{where}"
        return f"SELECT action, COUNT(*) AS count FROM {ALLOWED_TABLE}{where} GROUP BY action ORDER BY count DESC"

    # "average / avg latency or confidence"
    if "latency" in q and any(k in q for k in ("avg", "average", "mean")):
        where = f" WHERE {where_time}" if where_time else ""
        return f"SELECT avg(latency_ms) AS avg_latency_ms FROM {ALLOWED_TABLE}{where}"
    if "confidence" in q and any(k in q for k in ("avg", "average", "mean")):
        where = f" WHERE {where_time}" if where_time else ""
        return f"SELECT action, avg(confidence) AS avg_confidence FROM {ALLOWED_TABLE}{where} GROUP BY action"

    # "per hour / over time / trend"
    if any(k in q for k in ("per hour", "over time", "trend", "by hour", "hourly")):
        where = f" WHERE {where_time}" if where_time else ""
        return (
            f"SELECT date_trunc('hour', ts) AS hour, action, COUNT(*) AS count "
            f"FROM {ALLOWED_TABLE}{where} GROUP BY hour, action ORDER BY hour"
        )

    # default: recent events (optionally filtered)
    clauses = []
    if action:
        clauses.append(f"action = '{action}'")
    if where_time:
        clauses.append(where_time)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return (
        f"SELECT ts, subject_id, action, confidence FROM {ALLOWED_TABLE}{where} "
        f"ORDER BY ts DESC LIMIT 50"
    )


def _extract_window(q: str) -> str | None:
    m = re.search(r"(\d+)\s*(second|minute|hour|day|week)s?", q)
    if m:
        return f"{m.group(1)} {m.group(2)}s"
    if "today" in q:
        return "1 day"
    if "last hour" in q or "past hour" in q:
        return "1 hour"
    if "this week" in q or "last week" in q:
        return "7 days"
    return None


# ── orchestration ─────────────────────────────────────────────────────────────
_LLM_SYSTEM = (
    "You are a careful analytics assistant. Convert the user's question into a "
    "single read-only PostgreSQL SELECT over the given schema. Return ONLY the SQL, "
    "no prose, no comments, no semicolon. Never modify data."
)


async def generate_sql(question: str, provider: LLMProvider) -> SQLGeneration:
    if provider.is_offline:
        raw, prov = rule_based_sql(question), "offline"
    else:
        prompt = f"{SCHEMA_DESCRIPTION}\n\nQuestion: {question}\n\nSQL:"
        result = await provider.complete(prompt, system=_LLM_SYSTEM, temperature=0.0, max_tokens=300)
        raw, prov = result.text, result.provider

    try:
        safe = validate_select(raw)
        return SQLGeneration(question=question, sql=safe, provider=prov, valid=True)
    except UnsafeQueryError as e:
        return SQLGeneration(question=question, sql=raw, provider=prov, valid=False, error=str(e))
