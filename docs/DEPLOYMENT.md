# Deployment (100% free tier)

Goal: a public live demo at **$0/month**. Compute runs on Railway, while the
database and Redis are offloaded to managed free tiers so they don't burn
Railway's trial credit. The trained model is baked into the Docker image, so the
deployed app works the moment it boots.

```
Railway (Docker: API)  ──►  Supabase (Postgres + pgvector)
                       └──►  Upstash (Redis, serverless)
                       └──►  Groq API (free tier)  [optional]
```

> Everything below also works with **Render** (use [`render.yaml`](../render.yaml))
> or **Fly.io** instead of Railway. Swap Supabase for **Neon** freely — both ship
> pgvector. The app auto-detects whether TimescaleDB/pgvector are available.

---

## 1. Database — Supabase (free)

1. Create a project at supabase.com (free tier: 500 MB Postgres, pgvector included).
2. **Database → Connection string → URI**. Copy it; it looks like:
   `postgresql://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres`
3. Convert the scheme to the async driver this app uses:
   ```
   postgresql+asyncpg://postgres:[PASSWORD]@db.xxxx.supabase.co:5432/postgres
   ```
   This is your `WIFIPOSE_DATABASE_URL`. The schema is created automatically on
   first boot. (TimescaleDB isn't on Supabase — the app detects this and uses the
   BRIN/date_trunc path. pgvector **is** available and will be used.)

## 2. Redis — Upstash (free)

1. Create a database at upstash.com (free tier: 10k commands/day, serverless).
2. Copy the **Redis URL** (TLS): `rediss://default:[TOKEN]@xxxx.upstash.io:6379`.
   This is your `WIFIPOSE_REDIS_URL`.

## 3. LLM key — Groq (free, optional)

1. Get a key at console.groq.com/keys (free dev tier, no credit card; serves
   Llama 3.x at very low latency).
2. Set `WIFIPOSE_LLM_PROVIDER=groq` and `WIFIPOSE_LLM_API_KEY=[KEY]`.

> Skip this entirely and leave `WIFIPOSE_LLM_PROVIDER=offline` — NL→SQL and
> summaries still work via the deterministic fallback. Gemini is also supported
> as a drop-in (`WIFIPOSE_LLM_PROVIDER=gemini`).

## 4. Deploy on Railway

1. Push this repo to GitHub.
2. railway.app → **New Project → Deploy from GitHub repo**. Railway detects the
   [`Dockerfile`](../Dockerfile) (and [`railway.json`](../railway.json), which sets
   the `/health` healthcheck). The container's `CMD` binds uvicorn to Railway's
   injected `$PORT` via a shell, so no custom start command is needed.
3. In the service **Variables**, add:

   | Variable | Value |
   |---|---|
   | `WIFIPOSE_DATABASE_URL` | the Supabase async URI from step 1 |
   | `WIFIPOSE_REDIS_URL` | the Upstash URL from step 2 |
   | `WIFIPOSE_LLM_PROVIDER` | `groq` (or `offline`) |
   | `WIFIPOSE_LLM_API_KEY` | your Groq key (omit if offline) |
   | `WIFIPOSE_ENV` | `production` |
   | `WIFIPOSE_LOG_JSON` | `true` |

   Railway injects `PORT` automatically; the container respects it.
4. Deploy. When it's live, open the service URL — the React UI is served by the
   same process; `/docs` has the API; `/metrics` is Prometheus-formatted.

## 5. Seed demo data (so analytics/LLM have something to show)

The deployed app has the model but an empty DB. Seed it from your laptop against
the cloud DB:

```bash
export WIFIPOSE_DATABASE_URL="postgresql+asyncpg://postgres:[PWD]@db.xxxx.supabase.co:5432/postgres"
python scripts/seed_demo.py --hours 6 --subjects 4
```

Or run the streaming pipeline continuously by deploying a second Railway service
from the same repo with start command `wifipose-produce`, plus a third with
`wifipose-consume` (both need `WIFIPOSE_REDIS_URL`; the consumer also needs
`WIFIPOSE_DATABASE_URL`).

---

## Optional: separate the frontend (Vercel/Netlify)

The frontend is served by FastAPI by default. To host it separately:

```bash
cd frontend && npm install && npm run build   # → frontend/dist
```

Deploy `frontend/dist` to Vercel/Netlify and point the API base at your Railway
URL (and set `WIFIPOSE_CORS_ORIGINS` to include the frontend origin).

---

## Free-tier limits to keep in mind

| Service | Free limit | Mitigation in this app |
|---|---|---|
| Supabase | 500 MB, pauses after 7 days idle | events are small JSON rows; `seed_demo` is modest |
| Upstash | 10k cmds/day | cache TTL + stream `MAXLEN` cap usage; offline demo needs no Redis |
| Groq | free dev tier, per-minute/day rate limits | offline fallback for everything; LLM only on explicit queries |
| Railway | trial credit, then hobby | DB/Redis offloaded; single small CPU service |
| Render (alt) | web service sleeps when idle | fast cold start (no torch in runtime image) |

---

## Smoke test a live deploy

```bash
BASE=https://your-app.up.railway.app
curl $BASE/health
curl $BASE/ready                                  # shows model_loaded + db capabilities
curl -X POST $BASE/api/v1/simulate -H 'content-type: application/json' -d '{"action":2}'
curl -X POST $BASE/api/v1/query -H 'content-type: application/json' \
     -d '{"question":"count events by action"}'
```
