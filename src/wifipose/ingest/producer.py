"""Synthetic CSI producer.

Stands in for real WiFi NIC capture: emits CSI windows for N simulated subjects
onto a Redis Stream (`XADD`). Decoupling capture from inference via a stream is
the core "systems" pattern — it gives durability, back-pressure, replay, and
horizontal worker scaling for free.

Run:  wifipose-produce        (uses WIFIPOSE_* env / .env)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal

import numpy as np

from wifipose.cache.redis_client import close_redis, get_redis
from wifipose.config import get_settings
from wifipose.logging_config import get_logger
from wifipose.ml.simulator import CSISimulator

log = get_logger("producer")


async def run_producer(rate_hz: float | None = None, subjects: int | None = None, maxlen: int = 10000) -> None:
    settings = get_settings()
    rate_hz = rate_hz or settings.producer_rate_hz
    subjects = subjects or settings.producer_subjects
    interval = 1.0 / max(rate_hz, 0.5)

    redis = get_redis()
    sim = CSISimulator()
    sims = {f"subject-{i}": np.random.default_rng(i) for i in range(subjects)}

    stop = asyncio.Event()
    _install_signal_handlers(stop)
    log.info("producer_started", rate_hz=rate_hz, subjects=subjects, stream=settings.stream_key)

    frame = 0
    while not stop.is_set():
        for subject_id, rng in sims.items():
            sample = sim.sample(rng=rng)
            await redis.xadd(
                settings.stream_key,
                {
                    "subject_id": subject_id,
                    "csi": json.dumps(sample.csi.tolist()),
                    "ground_truth": sample.action_name,
                    "frame": frame,
                },
                maxlen=maxlen,
                approximate=True,
            )
        frame += 1
        if frame % 50 == 0:
            log.info("produced", frames=frame, subjects=subjects)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

    await close_redis()
    log.info("producer_stopped", frames=frame)


def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Produce synthetic CSI frames to Redis Stream.")
    p.add_argument("--rate-hz", type=float, default=None)
    p.add_argument("--subjects", type=int, default=None)
    a = p.parse_args()
    asyncio.run(run_producer(rate_hz=a.rate_hz, subjects=a.subjects))


if __name__ == "__main__":
    main()
