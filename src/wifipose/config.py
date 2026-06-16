"""Centralized, validated configuration (pydantic-settings).

Every setting has a default so the app boots with zero env vars. All env vars
are prefixed `WIFIPOSE_`. This single object is the seam that lets the same code
run on local Docker (TimescaleDB + Redis) and free-tier cloud (Supabase/Neon +
Upstash) — only the connection strings change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wifipose import NUM_FEATURES, SEQ_LEN


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WIFIPOSE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # Model / inference
    model_path: str = "checkpoints/wifipose.onnx"
    seq_len: int = SEQ_LEN
    num_features: int = NUM_FEATURES
    onnx_threads: int = 1

    # Database
    database_url: str = "postgresql+asyncpg://wifipose:wifipose@localhost:5432/wifipose"
    db_enable_timescale: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    stream_key: str = "csi:frames"
    consumer_group: str = "inference"
    prediction_cache_ttl: int = 30

    # LLM
    llm_provider: str = "offline"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout: int = 20

    # Demo / ingest
    producer_rate_hz: float = 10.0
    producer_subjects: int = 3

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:8000"

    @field_validator("cors_origins")
    @classmethod
    def _split_origins(cls, v: str) -> str:
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
