"""Seed the database with demo pose events so analytics/LLM queries have data.

Generates a few hours of realistic events across several subjects using the
simulator + the exported ONNX model. Safe to run repeatedly.

Usage:  python scripts/seed_demo.py --hours 6 --subjects 4
"""

from __future__ import annotations

import argparse
import asyncio
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from wifipose.config import get_settings
from wifipose.db.models import PoseEventIn
from wifipose.db.repository import PoseRepository
from wifipose.db.session import dispose_engine, init_db
from wifipose.ml.engine import PoseEstimator
from wifipose.ml.simulator import CSISimulator


async def seed(hours: int, subjects: int, events_per_hour: int) -> None:
    settings = get_settings()
    await init_db()
    repo = PoseRepository()
    estimator = PoseEstimator(settings.model_path, intra_op_threads=settings.onnx_threads)
    sim = CSISimulator(seed=7)

    now = datetime.now(timezone.utc)
    total = hours * events_per_hour
    written = 0
    for i in range(total):
        # spread timestamps backwards over the window
        ts = now - timedelta(hours=hours) + timedelta(hours=hours * i / total)
        # bias action distribution to look realistic (mostly Normal)
        action = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.15, 0.05])[0]
        sample = sim.sample(action=action)
        pred = estimator.predict(sample.csi)
        mean_conf = float(np.mean(pred.confidence)) if pred.confidence else 0.0
        event = PoseEventIn(
            subject_id=f"subject-{i % subjects}",
            action_index=pred.action_index,
            action=pred.action_name,
            confidence=mean_conf,
            action_probs=pred.action_probs,
            uv_coords=pred.uv_coords,
            embedding=PoseEventIn.embedding_from_uv(pred.uv_coords),
            latency_ms=pred.latency_ms,
            source="seed",
            ts=ts,
        )
        await repo.insert_event(event)
        written += 1
        if written % 50 == 0:
            print(f"  seeded {written}/{total}")

    print(f"Done. Seeded {written} events across {subjects} subjects over {hours}h.")
    await dispose_engine()


def main() -> None:
    p = argparse.ArgumentParser(description="Seed demo pose events.")
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--subjects", type=int, default=4)
    p.add_argument("--per-hour", type=int, default=60)
    a = p.parse_args()
    asyncio.run(seed(a.hours, a.subjects, a.per_hour))


if __name__ == "__main__":
    main()
