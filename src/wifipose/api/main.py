"""FastAPI application factory and lifecycle wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from wifipose import __version__
from wifipose.api.deps import load_estimator
from wifipose.api.routers import analytics, health, inference, ingest, query, stream
from wifipose.cache.redis_client import close_redis
from wifipose.config import get_settings
from wifipose.db.session import dispose_engine, init_db
from wifipose.logging_config import configure_logging, get_logger

log = get_logger("app")
_FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    log.info("starting", env=settings.env, version=__version__)

    # DB schema bootstrap is best-effort: the analytics features degrade, but the
    # inference + demo endpoints must come up even if no database is reachable.
    try:
        await init_db()
    except Exception as e:  # noqa: BLE001
        log.warning("db_init_failed", error=str(e))

    load_estimator()
    yield

    await close_redis()
    await dispose_engine()
    log.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="WiFi Pose Intelligence Platform",
        version=__version__,
        description="Camera-free human pose & action intelligence from WiFi CSI.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (health, inference, stream, analytics, query, ingest):
        app.include_router(module.router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React app if present; otherwise redirect root to docs."""
    if _FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):  # noqa: ARG001 - SPA client routing
            index = _FRONTEND_DIST / "index.html"
            return FileResponse(index)
    else:

        @app.get("/", include_in_schema=False)
        async def root():
            return RedirectResponse(url="/docs")


app = create_app()
