"""API tests — full app, no external services (deps overridden with fakes)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeRepository

from wifipose import ACTION_NAMES, NUM_PARTS
from wifipose.api.deps import get_estimator, get_repository
from wifipose.api.main import create_app
from wifipose.ml.engine import PosePrediction


class FakeEstimator:
    def predict(self, csi: np.ndarray) -> PosePrediction:
        return PosePrediction(
            uv_coords=[[0.1, 0.2]] * NUM_PARTS,
            confidence=[0.9] * NUM_PARTS,
            action_index=2,
            action_name="Running",
            action_probs={n: (0.7 if n == "Running" else 0.1) for n in ACTION_NAMES},
            latency_ms=4.2,
        )

    @staticmethod
    def validate_window(csi):
        return PosePrediction.__init__  # not used in these paths


@pytest.fixture
def client(fake_repo: FakeRepository):
    app = create_app()
    app.dependency_overrides[get_estimator] = lambda: FakeEstimator()
    app.dependency_overrides[get_repository] = lambda: fake_repo
    # get_llm stays as the real offline provider (deterministic, no network)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert "llm_provider" in r.json()


def test_simulate_runs_inference(client):
    r = client.post("/api/v1/simulate", json={"action": 2, "persist": True})
    assert r.status_code == 200
    body = r.json()
    assert body["action_name"] == "Running"
    assert body["ground_truth_action"] == "Running"
    assert "event_id" in body  # persisted via fake repo


def test_events_listing_after_persist(client):
    client.post("/api/v1/simulate", json={"action": 1, "persist": True})
    r = client.get("/api/v1/events?limit=10")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_nl_query_offline(client):
    r = client.post("/api/v1/query", json={"question": "count events by action", "execute": True})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert "group by action" in body["sql"].lower()
    assert body["provider"] == "offline"


def test_nl_query_blocks_unsafe_intent(client):
    # Even adversarial phrasing only ever yields a SELECT from the rule engine.
    r = client.post("/api/v1/query", json={"question": "drop all the tables now", "execute": False})
    assert r.status_code == 200
    assert r.json()["sql"].lower().startswith("select")


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "wifipose_inferences_total" in r.text


def test_query_schema(client):
    r = client.get("/api/v1/query/schema")
    assert r.status_code == 200
    assert "pose_events" in r.json()["schema"]
