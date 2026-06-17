"""Device registration + authenticated CSI ingestion (Phase 2)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeDeviceRepository, FakeRedis

import wifipose.api.routers.ingest as ingest_module
from wifipose.api.deps import get_device_repository
from wifipose.api.main import create_app


@pytest.fixture
def client(monkeypatch):
    app = create_app()
    devices = FakeDeviceRepository()  # one shared instance across register + ingest
    app.dependency_overrides[get_device_repository] = lambda: devices
    # capture XADD instead of hitting real Redis
    fake_redis = FakeRedis()
    monkeypatch.setattr(ingest_module, "get_redis", lambda: fake_redis)
    with TestClient(app) as c:
        c.fake_redis = fake_redis
        yield c


def test_register_returns_key_once(client):
    r = client.post("/api/v1/devices", json={"name": "my-esp32"})
    assert r.status_code == 201
    body = r.json()
    assert body["api_key"].startswith("wfp_")
    assert body["id"].startswith("dev_")
    assert body["csi_format"] == "complex_interleaved"


def test_ingest_requires_valid_key(client):
    window = {"packets": [[0.1] * 104] * 30, "format": "amp_phase"}
    # no key -> 401
    assert client.post("/api/v1/ingest/window", json=window).status_code == 401
    # bogus key -> 401
    bad = client.post("/api/v1/ingest/window", json=window, headers={"X-API-Key": "wfp_nope"})
    assert bad.status_code == 401


def test_register_then_ingest_emits_to_stream(client):
    key = client.post("/api/v1/devices", json={"name": "d"}).json()["api_key"]
    window = {"packets": [[0.2] * 104] * 30, "format": "amp_phase", "subject_id": "alice"}
    r = client.post("/api/v1/ingest/window", json=window, headers={"X-API-Key": key})
    assert r.status_code == 200
    assert r.json()["accepted"] is True
    # one window was queued onto the inference stream, tagged with the device id
    assert len(client.fake_redis.added) == 1
    _stream, fields = client.fake_redis.added[0]
    assert fields["subject_id"] == "alice"
    assert fields["device_id"].startswith("dev_")
    assert fields["source"] == "device"


def test_whoami(client):
    key = client.post("/api/v1/devices", json={"name": "d"}).json()["api_key"]
    r = client.get("/api/v1/devices/me", headers={"X-API-Key": key})
    assert r.status_code == 200
    assert r.json()["name"] == "d"
