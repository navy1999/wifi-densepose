"""Domain models shared between the repository and the API layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from wifipose import NUM_PARTS


class PoseEventIn(BaseModel):
    """A single inference result ready to persist."""

    subject_id: str
    action_index: int
    action: str
    confidence: float
    action_probs: dict[str, float]
    uv_coords: list[list[float]] = Field(..., description="[24][2] UV body-part coords")
    embedding: list[float] = Field(..., description="flattened pose feature vector (48-dim)")
    latency_ms: float
    source: str = "stream"
    ts: datetime | None = None

    @staticmethod
    def embedding_from_uv(uv_coords: list[list[float]]) -> list[float]:
        """Pose embedding = flattened UV coords (48-dim). Cheap, dependency-free,
        and good enough for cosine similarity / pgvector nearest-neighbour."""
        flat = [float(v) for pair in uv_coords[:NUM_PARTS] for v in pair[:2]]
        # pad to fixed length so the vector column width is stable
        flat += [0.0] * (NUM_PARTS * 2 - len(flat))
        return flat[: NUM_PARTS * 2]


class PoseEventOut(BaseModel):
    id: int
    ts: datetime
    subject_id: str
    action_index: int
    action: str
    confidence: float
    action_probs: dict[str, float]
    uv_coords: list[list[float]]
    latency_ms: float
    source: str


class SimilarPose(PoseEventOut):
    similarity: float


class ActionBucket(BaseModel):
    bucket: datetime
    action: str
    count: int
