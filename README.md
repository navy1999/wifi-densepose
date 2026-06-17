# WiFi Pose Intelligence Platform

> Camera-free human **pose & action sensing from WiFi signals** — wrapped in a production ML system: streaming ingest, ONNX serving, a time-series database with vector search, an LLM analytics layer, and a live demo.

[![CI](https://github.com/USER/wifi-densepose/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

WiFi routers continuously measure **Channel State Information (CSI)** — how the radio signal is distorted by everything in a room, including people. A neural network can turn those distortions into a human **pose skeleton** and an **action label**, *through walls and in the dark, with no camera*. This repo takes that research idea (originally a Colab notebook) and builds the **engineering system a real product would need around it**.

The model is the easy 10%. This repo is the other 90%: serving, data, streaming, observability, deployment, and an LLM query layer — all runnable on **free tiers** and on a laptop with **zero external dependencies**.

---

## Why this exists (the engineering story)

Three things a reviewer can verify in minutes:

| Claim | Evidence | Where |
|---|---|---|
| **6.6× faster inference** via PyTorch→ONNX + caching | p50 **2.5 ms** (ONNX) vs 16.8 ms (PyTorch eager), CPU-only | [`benchmark.py`](src/wifipose/ml/benchmark.py), `checkpoints/benchmark.json` |
| **LLM analytics over a time-series DB** | natural language → **safety-validated SQL** → results | [`nl_to_sql.py`](src/wifipose/llm/nl_to_sql.py) |
| **Runs anywhere, breaks nowhere** | CSI **simulator** + **offline LLM** fallback → no hardware, no API key, no data download required | [`simulator.py`](src/wifipose/ml/simulator.py) |

It also runs **identically** on local Docker (TimescaleDB + Redis + Grafana) and on **free-tier cloud** (Supabase + Upstash + Railway) — the only thing that changes is two connection strings.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   WiFi CSI (or          │  FastAPI service                             │
   synthetic simulator)  │  ┌────────────┐  REST /predict /simulate     │
        │                │  │ ONNX Engine│  WS  /ws  (live skeletons)    │
        ▼                │  │ (CPU, 2.5ms)│ ───────────────► React UI    │
  ┌───────────┐  XADD    │  └────────────┘                              │
  │ Producer  │ ───────► Redis Stream ──► ┌──────────┐  publish          │
  └───────────┘          │   (consumer    │  Worker  │ ─► pub/sub ──► WS  │
                         │    group)      │  pool xN │                   │
                         │                └────┬─────┘                   │
                         │                     │ insert                  │
                         │   ┌─────────────────▼──────────────────────┐  │
                         │   │ TimescaleDB / Postgres  (+ pgvector)    │  │
                         │   │  hypertable pose_events · ANN pose search│ │
                         │   └─────────────────┬──────────────────────┘  │
                         │   LLM layer ────────┘  NL→SQL · summaries      │
                         │   (Groq / Gemini / offline)                   │
                         └───────────────┬──────────────────────────────┘
                                         │ /metrics
                                  Prometheus → Grafana
```

Full write-up: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## The model

Faithfully reproduced from the source notebook ([`src/wifipose/ml/model.py`](src/wifipose/ml/model.py)):

```
CSI window [B, 30, 104]                     # 30 timesteps × (52 subcarriers × amp+phase)
   → CSIEncoder            (per-step MLP → 256-d embedding)
   → DensePoseNet          (4-layer Transformer → 24 UV body-parts + confidence)
   → ActionRecognitionNet  (temporal Conv1d → Normal / Loitering / Running / Aggressive)
```

**3.34M params (~13 MB)**. Tiny by design — runs in single-digit milliseconds on CPU, so inference needs **no GPU and no paid compute**.

---

## Quickstart

### Option A — zero dependencies (60 seconds)

No Docker, no database, no API keys. Trains a demo model on the simulator, serves it, opens the live demo.

```bash
pip install -e ".[ml]"                 # ml extras include torch (train + export only)
make checkpoint                        # train on simulated CSI  → checkpoints/wifipose.pth
make export                            # PyTorch → ONNX (+ parity check)
make benchmark                         # prints the 6.6× speedup table
make api                               # http://localhost:8000  (Swagger at /docs)
```

Then in another terminal, see the model run:

```bash
curl -X POST localhost:8000/api/v1/simulate -H "content-type: application/json" -d '{"action":2}'
```

> Windows (no `make`): run the underlying commands from the [Makefile](Makefile), e.g. `wifipose-checkpoint`, `wifipose-export`, `uvicorn wifipose.api.main:app --reload`.

### Option B — full stack (Docker)

Brings up TimescaleDB (+pgvector), Redis, the API, **2 worker replicas**, a CSI producer, Prometheus and Grafana. The image trains + exports the model at build time.

```bash
docker compose up --build
# app:      http://localhost:8000
# grafana:  http://localhost:3000   (anonymous viewer enabled)
# metrics:  http://localhost:8000/metrics
```

Seed some history so the analytics/LLM features have data:

```bash
make seed         # or: python scripts/seed_demo.py --hours 6 --subjects 4
```

### Frontend dev

```bash
make frontend-dev     # Vite dev server on :5173, proxies /api → :8000
```

---

## Ask the data in plain English

The `/api/v1/query` endpoint turns a question into a **read-only, validated** SQL query over the pose event log:

```bash
curl -X POST localhost:8000/api/v1/query -H "content-type: application/json" \
  -d '{"question":"how many running events in the last 2 hours?"}'
```
```json
{
  "question": "how many running events in the last 2 hours?",
  "sql": "SELECT COUNT(*) AS count FROM pose_events WHERE action = 'Running' AND ts >= now() - interval '2 hours'",
  "provider": "offline",
  "valid": true,
  "rows": [{"count": 37}]
}
```

Safety is enforced regardless of what the LLM emits: single statement, `SELECT`/`WITH` only, table allowlist, no comments, forced `LIMIT`, executed in a `READ ONLY` transaction with a statement timeout. See [`nl_to_sql.py`](src/wifipose/llm/nl_to_sql.py) and its [tests](tests/test_nl_to_sql.py).

With **no API key**, a deterministic rule-based generator handles the common intents. Set `WIFIPOSE_LLM_PROVIDER=groq` (free tier) + `WIFIPOSE_LLM_API_KEY=...` to use a real LLM for arbitrary questions.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/predict` | Inference on a caller-supplied CSI window |
| `POST` | `/api/v1/simulate` | Generate a synthetic window + infer (keyless demo) |
| `WS` | `/api/v1/ws` | Live skeleton stream (`?mode=sim` or `live`) |
| `GET` | `/api/v1/events` | Recent pose events (filterable) |
| `GET` | `/api/v1/stats/histogram` · `/stats/timeseries` | Action aggregates (time-bucketed) |
| `GET` | `/api/v1/similar/{event_id}` | Nearest-neighbour pose search (pgvector / cosine) |
| `GET` | `/api/v1/summary` | LLM activity briefing |
| `POST` | `/api/v1/query` | Natural-language → SQL analytics |
| `GET` | `/health` · `/ready` · `/metrics` | Ops endpoints |

---

## Tech stack

| Concern | Choice | Free-tier path |
|---|---|---|
| Model serving | **ONNX Runtime** (CPU) | in-process, $0 |
| API | **FastAPI** + WebSockets | Railway / Render |
| Streaming | **Redis Streams** consumer groups | Upstash (serverless) |
| Database | **TimescaleDB** → degrades to plain **Postgres** | Supabase / Neon |
| Vector search | **pgvector** → cosine fallback | Supabase / Neon |
| LLM | provider-agnostic: **Groq / Gemini / offline** | Groq free tier |
| Observability | **Prometheus + Grafana** | local / Grafana Cloud |
| Frontend | **React + Vite + Canvas** | served by FastAPI / Vercel |
| CI | **GitHub Actions** (lint, tests, ONNX parity, Docker) | free |

---

## Project layout

```
src/wifipose/
  ml/         model · simulator · ONNX export · benchmark · training · serving engine
  api/        FastAPI app, routers (inference/stream/analytics/query/health), metrics
  db/         async SQLAlchemy, schema.sql (Timescale+pgvector), raw-SQL repository
  llm/        provider abstraction, NL→SQL (+ safety validator), summarizer
  cache/      Redis client (prediction cache + stream)
  ingest/     synthetic CSI producer  →  Redis Stream
  worker/     stream consumer  →  inference  →  DB  →  pub/sub
frontend/     React live skeleton viewer + NL query panel
infra/        Prometheus + Grafana provisioning
tests/        36 tests — model, simulator, ONNX parity, SQL safety, API
docs/         ARCHITECTURE.md · DEPLOYMENT.md
```

---

## Deploy it (free tier)

Step-by-step for **Supabase + Upstash + Groq + Railway** in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. Short version: create the three free accounts, paste three connection strings as env vars, `git push` to Railway. The model is baked into the image, so the deployed demo works immediately.

---

## Credits & scope

- Inspired by the **"DensePose From WiFi"** research direction (CMU). The original model was built and trained in Colab — that notebook is preserved at [`notebooks/model-training-colab.ipynb`](notebooks/model-training-colab.ipynb), and [`model.py`](src/wifipose/ml/model.py) is a faithful reproduction of its architecture.
- Synthetic CSI powers the runnable demo; the real-data training path ([`dataset.py`](src/wifipose/ml/dataset.py), `wifipose-train`) mirrors the notebook against the Kaggle `wifipose-dataset` and `csi-human-in-wifi` datasets.
- This project is about the **systems engineering around an ML model**, not a novel sensing result.
