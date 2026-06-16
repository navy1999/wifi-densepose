"""Async engine, session factory, schema bootstrap, and capability detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from wifipose.config import get_settings
from wifipose.logging_config import get_logger

log = get_logger("db")

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_STATEMENT_SEP = "-- @@STATEMENT@@"


@dataclass
class Capabilities:
    """What the connected database actually supports (detected at startup)."""

    timescaledb: bool = False
    pgvector: bool = False


_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_caps = Capabilities()


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            future=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _sessionmaker


def capabilities() -> Capabilities:
    return _caps


async def init_db() -> Capabilities:
    """Apply schema.sql idempotently and detect extension support."""
    settings = get_settings()
    statements = [s.strip() for s in _SCHEMA_PATH.read_text().split(_STATEMENT_SEP)]
    statements = [s for s in statements if s and not s.startswith("--") or "\n" in s]

    engine = get_engine()
    async with engine.begin() as conn:
        for stmt in statements:
            body = "\n".join(ln for ln in stmt.splitlines() if not ln.strip().startswith("--"))
            if not body.strip():
                continue
            # TimescaleDB extension creation can be disabled via config.
            if "CREATE EXTENSION IF NOT EXISTS timescaledb" in body and not settings.db_enable_timescale:
                continue
            await conn.exec_driver_sql(body)

        rows = (await conn.exec_driver_sql("SELECT extname FROM pg_extension")).fetchall()
        installed = {r[0] for r in rows}
        _caps.timescaledb = "timescaledb" in installed
        _caps.pgvector = "vector" in installed

    log.info("db_initialized", timescaledb=_caps.timescaledb, pgvector=_caps.pgvector)
    return _caps


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
