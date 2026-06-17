"""Prometheus metrics — the observability surface scraped by Grafana."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_LATENCY = Histogram(
    "wifipose_inference_latency_ms",
    "Model inference latency in milliseconds",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)
INFERENCE_TOTAL = Counter(
    "wifipose_inferences_total", "Total inference requests", ["source", "action"]
)
CACHE_HITS = Counter("wifipose_cache_hits_total", "Prediction cache hits")
CACHE_MISSES = Counter("wifipose_cache_misses_total", "Prediction cache misses")
EVENTS_PERSISTED = Counter("wifipose_events_persisted_total", "Pose events written to DB")
WS_CONNECTIONS = Gauge("wifipose_ws_connections", "Active WebSocket connections")
LLM_QUERIES = Counter("wifipose_llm_queries_total", "NL analytics queries", ["provider", "valid"])
INGEST_FRAMES = Counter(
    "wifipose_ingest_frames_total", "CSI windows ingested from devices", ["device"]
)
