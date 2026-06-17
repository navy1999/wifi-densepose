-- WiFi Pose Intelligence Platform — schema (idempotent, portable).
-- Statements are separated by the sentinel below so the async init code can run
-- them one at a time (asyncpg executes a single statement per call).
-- Works on: TimescaleDB, plain Postgres 14+, Supabase, Neon.

-- @@STATEMENT@@
-- TimescaleDB is optional. Swallow the error on hosts that don't ship it.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'timescaledb unavailable (%); continuing on plain Postgres', SQLERRM;
END $$;

-- @@STATEMENT@@
-- pgvector is optional. Available by default on Supabase/Neon; absent on vanilla PG.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector unavailable (%); similarity search will use app-side cosine', SQLERRM;
END $$;

-- @@STATEMENT@@
CREATE TABLE IF NOT EXISTS pose_events (
    ts           TIMESTAMPTZ      NOT NULL DEFAULT now(),
    id           BIGINT GENERATED ALWAYS AS IDENTITY,
    device_id    TEXT             NOT NULL DEFAULT 'demo',
    subject_id   TEXT             NOT NULL,
    action_index SMALLINT         NOT NULL,
    action       TEXT             NOT NULL,
    confidence   REAL             NOT NULL,
    action_probs JSONB            NOT NULL,
    uv_coords    JSONB            NOT NULL,
    embedding    REAL[]           NOT NULL,
    latency_ms   REAL             NOT NULL,
    source       TEXT             NOT NULL DEFAULT 'stream',
    PRIMARY KEY (ts, id)
);

-- @@STATEMENT@@
-- device_id added in Phase 2; backfill-safe for tables created before then.
ALTER TABLE pose_events ADD COLUMN IF NOT EXISTS device_id TEXT NOT NULL DEFAULT 'demo';

-- @@STATEMENT@@
-- Registered capture devices (one per user's ESP32 / Pi / NIC). The API key is
-- stored only as a SHA-256 hash; the plaintext is shown once at registration.
CREATE TABLE IF NOT EXISTS devices (
    id            TEXT        PRIMARY KEY,
    name          TEXT        NOT NULL,
    owner         TEXT        NOT NULL DEFAULT 'anonymous',
    api_key_hash  TEXT        NOT NULL UNIQUE,
    csi_format    TEXT        NOT NULL DEFAULT 'complex_interleaved',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen     TIMESTAMPTZ
);

-- @@STATEMENT@@
-- Promote to a hypertable when TimescaleDB is present (no-op otherwise).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('pose_events', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
    END IF;
END $$;

-- @@STATEMENT@@
-- BRIN index on time: tiny and effective for append-only time-series scans,
-- and works on every Postgres (a portable substitute for hypertable chunking).
CREATE INDEX IF NOT EXISTS pose_events_ts_brin ON pose_events USING brin (ts);

-- @@STATEMENT@@
CREATE INDEX IF NOT EXISTS pose_events_action_ts ON pose_events (action, ts DESC);

-- @@STATEMENT@@
CREATE INDEX IF NOT EXISTS pose_events_subject_ts ON pose_events (subject_id, ts DESC);

-- @@STATEMENT@@
CREATE INDEX IF NOT EXISTS pose_events_device_ts ON pose_events (device_id, ts DESC);

-- @@STATEMENT@@
-- When pgvector exists, add a typed vector column + ANN index for fast
-- nearest-neighbour pose search. Backfilled from the portable REAL[] column.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        ALTER TABLE pose_events ADD COLUMN IF NOT EXISTS embedding_vec vector(48);
        BEGIN
            CREATE INDEX IF NOT EXISTS pose_events_embedding_ivf
                ON pose_events USING ivfflat (embedding_vec vector_cosine_ops) WITH (lists = 50);
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'ivfflat index skipped (%).', SQLERRM;
        END;
    END IF;
END $$;
