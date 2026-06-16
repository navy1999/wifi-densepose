# Architecture

This document explains *why* the system is shaped the way it is — the decisions a
reviewer would ask about.

## 1. Data contract (defined once)

The CSI window shape is the spine of the whole system and is declared in exactly
one place ([`wifipose/__init__.py`](../src/wifipose/__init__.py)):

```
SEQ_LEN = 30          # timesteps
NUM_FEATURES = 104    # 52 subcarriers × (amplitude + phase)
NUM_PARTS = 24        # DensePose UV body parts
NUM_CLASSES = 4       # Normal / Loitering / Running / Aggressive
```

Every component — model, simulator, ONNX engine, DB embedding width, validation —
imports these constants, so the contract can't drift between layers.

## 2. Why ONNX for serving (the optimization story)

The training code uses PyTorch; the **serving path never imports torch**. At build
time we export to ONNX and verify numerical parity (max abs diff ~1e-7) before the
artifact is accepted ([`export.py`](../src/wifipose/ml/export.py)).

| Backend | p50 | p99 | Notes |
|---|---|---|---|
| PyTorch eager (CPU, 1 thread) | 16.8 ms | 22.3 ms | full Python/autograd overhead |
| **ONNX Runtime (CPU, 1 thread)** | **2.5 ms** | **3.4 ms** | graph-optimized, **6.6× faster** |

Consequences:
- **No GPU needed** → fits free-tier CPU dynos.
- The runtime image excludes torch (~hundreds of MB saved), so cold starts on
  sleep-to-zero free hosts are fast.
- A **Redis prediction cache** keyed on a rounded CSI fingerprint dedupes
  identical/near-identical windows; cache hits skip inference entirely.

## 3. Streaming pipeline (decoupling capture from compute)

```
producer ──XADD──► Redis Stream ──XREADGROUP──► worker(s) ──► Postgres
                                                   └──publish──► pub/sub ──► WebSocket
```

A **Redis Stream + consumer group** sits between ingestion and inference. This is
the core systems pattern and it buys:

- **Back-pressure & durability** — bursts queue instead of dropping; `MAXLEN`
  caps memory.
- **Horizontal scale** — `docker-compose` runs **2 worker replicas**; Redis
  partitions messages across them. Add more workers, get more throughput, no code
  change.
- **At-least-once delivery** — messages are `XACK`-ed only after handling; a
  crashed worker's pending entries can be reclaimed.
- **Fan-out** — workers publish results to a pub/sub channel that the API's
  WebSocket tails, so every connected browser sees the same live stream.

## 4. Storage: time-series + vectors, with graceful degradation

One table, `pose_events`, defined in [`schema.sql`](../src/wifipose/db/schema.sql).
The schema is **idempotent and self-adapting**:

- If **TimescaleDB** is present → `pose_events` becomes a **hypertable** and
  aggregates use `time_bucket()`.
- If only plain **Postgres** is available (Supabase/Neon free tier) → it stays a
  normal table with a **BRIN index** on `ts` (cheap, effective for append-only
  time-series) and aggregates fall back to `date_trunc()`.
- If **pgvector** is present → a `vector(48)` column + **ivfflat ANN index** powers
  `/similar/{id}`; otherwise the repository computes cosine similarity in Python.

Capabilities are detected once at startup ([`session.py`](../src/wifipose/db/session.py))
and the repository branches on them. The pose **embedding** is just the flattened
24×2 UV coordinates — cheap, dependency-free, and meaningful for "find similar
poses".

The repository ([`repository.py`](../src/wifipose/db/repository.py)) uses raw
parameterized SQL on purpose: it makes the actual time-series queries visible and
keeps the same query surface honest for the LLM text-to-SQL feature.

## 5. LLM layer (useful, swappable, and safe)

`LLMProvider` is a one-method interface ([`base.py`](../src/wifipose/llm/base.py))
with implementations for **Gemini, Groq, OpenAI, Anthropic**, and an
**OfflineProvider**. `get_provider()` selects via config and **falls back to
offline** on any misconfiguration, so the app never hard-fails on a bad key.

Two features sit on top:

- **NL → SQL** ([`nl_to_sql.py`](../src/wifipose/llm/nl_to_sql.py)). Defense in
  depth: (1) the model is given only the table schema and told to emit one
  `SELECT`; (2) `validate_select()` enforces a strict allowlist — `SELECT`/`WITH`
  only, no DDL/DML keywords, no comments, single statement, table must be
  `pose_events`, mandatory `LIMIT`; (3) execution happens in a `READ ONLY`
  transaction with a 5s statement timeout. Even a validator bug cannot mutate
  data. Offline, a rule-based generator covers the common intents.
- **Summaries** ([`summarizer.py`](../src/wifipose/llm/summarizer.py)) with a
  deterministic extractive fallback.

## 6. Observability

`/metrics` exposes Prometheus counters/histograms (inference latency, cache
hit/miss, events persisted, WS connections, LLM query outcomes). Grafana is
auto-provisioned with a datasource + dashboard ([`infra/grafana`](../infra/grafana)).
`/health` (liveness) and `/ready` (reports model + DB capability state) support
orchestrator probes.

## 7. Config as the only environment seam

[`config.py`](../src/wifipose/config.py) (pydantic-settings) gives every value a
safe default, so the app boots with **zero env vars**. Local Docker vs free-tier
cloud differ only in `WIFIPOSE_DATABASE_URL`, `WIFIPOSE_REDIS_URL`, and the LLM
settings. Same image, same code, different strings.

## 8. Failure philosophy

The demo must never break in front of a recruiter:
- No model? Inference returns `503`; the rest of the API still serves.
- No DB? Schema init and persistence are best-effort; inference + simulate still work.
- No Redis? Cache ops swallow errors (treated as misses); the WebSocket `sim` mode
  is fully self-contained.
- No LLM key? Offline provider handles NL→SQL and summaries deterministically.

## 9. Deliberate non-goals

- **No Kubernetes.** Docker Compose is clearer for a portfolio system and matches
  the free-tier deployment target.
- **No model retraining claims.** The contribution is the system; the model is
  reproduced from the source notebook and trained briefly on synthetic data for a
  self-contained demo.
- **Synthetic CSI by default.** Real capture needs specific NIC hardware; the
  real-data training path exists ([`dataset.py`](../src/wifipose/ml/dataset.py))
  but is opt-in.
