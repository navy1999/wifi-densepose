"""Device API-key generation, hashing, and the auth dependency.

Keys look like `wfp_<token>` and are shown to the user exactly once at
registration. Only a SHA-256 hash is stored, so a database leak never exposes a
usable key. The same scheme guards both the HTTP and WebSocket ingest paths.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from wifipose.api.deps import get_device_repository

API_KEY_HEADER = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def new_device_id() -> str:
    return "dev_" + secrets.token_hex(8)


def generate_api_key() -> tuple[str, str]:
    """Return (plaintext_key, sha256_hash)."""
    plaintext = "wfp_" + secrets.token_urlsafe(32)
    return plaintext, hash_api_key(plaintext)


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


async def authenticate_device(api_key: str | None, repo) -> dict:
    """Validate a key against the devices table; raise 401 if invalid."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key (X-API-Key header).")
    device = await repo.get_by_key_hash(hash_api_key(api_key))
    if device is None:
        raise HTTPException(status_code=401, detail="Invalid device API key.")
    return device


async def require_device(
    api_key: str | None = Security(_api_key_scheme),
    repo=Depends(get_device_repository),
) -> dict:
    """FastAPI dependency for HTTP ingest routes."""
    return await authenticate_device(api_key, repo)
