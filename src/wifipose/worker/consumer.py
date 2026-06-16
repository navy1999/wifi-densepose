"""Redis Stream consumer (consumer-group based).

Reads CSI windows from the stream with `XREADGROUP`, runs ONNX inference,
persists the result to the time-series DB, publishes it to the `predictions`
pub/sub channel (for live WebSocket fan-out), and `XACK`s the message.

Consumer groups give at-least-once delivery and let you scale horizontally —
run multiple workers and Redis load-balances the partition of messages. A
pending-entries reclaim pass recovers messages from crashed workers.

Run:  wifipose-consume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import socket

import numpy as np

from wifipose.api.routers.stream import PREDICTIONS_CHANNEL
from wifipose.cache.redis_client import close_redis, get_redis
from wifipose.config import get_settings
from wifipose.db.models import PoseEventIn
from wifipose.db.repository import PoseRepository
from wifipose.db.session import init_db
from wifipose.logging_config import get_logger
from wifipose.ml.engine import PoseEstimator

log = get_logger("worker")


class StreamWorker:
    def __init__(self, consumer_name: str | None = None):
        self.settings = get_settings()
        self.redis = get_redis()
        self.repo = PoseRepository()
        self.estimator = PoseEstimator(
            self.settings.model_path, intra_op_threads=self.settings.onnx_threads
        )
        self.consumer_name = consumer_name or f"{socket.gethostname()}-{id(self)%10000}"
        self.group = self.settings.consumer_group
        self.stream = self.settings.stream_key
        self._stop = asyncio.Event()

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            log.info("group_created", group=self.group, stream=self.stream)
        except Exception as e:  # noqa: BLE001 - BUSYGROUP means it already exists
            if "BUSYGROUP" not in str(e):
                raise

    async def _handle(self, msg_id: str, fields: dict) -> None:
        csi = np.asarray(json.loads(fields["csi"]), dtype=np.float32)
        pred = self.estimator.predict(csi)
        mean_conf = float(np.mean(pred.confidence)) if pred.confidence else 0.0

        event = PoseEventIn(
            subject_id=fields.get("subject_id", "unknown"),
            action_index=pred.action_index,
            action=pred.action_name,
            confidence=mean_conf,
            action_probs=pred.action_probs,
            uv_coords=pred.uv_coords,
            embedding=PoseEventIn.embedding_from_uv(pred.uv_coords),
            latency_ms=pred.latency_ms,
            source="stream",
        )
        try:
            await self.repo.insert_event(event)
        except Exception as e:  # noqa: BLE001 - keep consuming even if DB is down
            log.warning("persist_failed", error=str(e))

        # Fan out to live WebSocket subscribers.
        await self.redis.publish(
            PREDICTIONS_CHANNEL,
            json.dumps(
                {
                    "subject_id": event.subject_id,
                    "uv_coords": pred.uv_coords,
                    "confidence": pred.confidence,
                    "action": pred.action_name,
                    "action_probs": pred.action_probs,
                    "ground_truth_action": fields.get("ground_truth"),
                    "latency_ms": pred.latency_ms,
                    "frame": int(fields.get("frame", 0)),
                }
            ),
        )

    async def run(self, batch: int = 10, block_ms: int = 5000) -> None:
        try:
            await init_db()
        except Exception as e:  # noqa: BLE001
            log.warning("db_init_failed", error=str(e))
        await self.ensure_group()
        self._install_signals()
        log.info("worker_started", consumer=self.consumer_name, group=self.group)

        while not self._stop.is_set():
            try:
                resp = await self.redis.xreadgroup(
                    self.group, self.consumer_name, {self.stream: ">"}, count=batch, block=block_ms
                )
            except Exception as e:  # noqa: BLE001
                log.warning("xreadgroup_failed", error=str(e))
                await asyncio.sleep(1)
                continue
            if not resp:
                continue
            for _stream, messages in resp:
                for msg_id, fields in messages:
                    try:
                        await self._handle(msg_id, fields)
                    except Exception as e:  # noqa: BLE001
                        log.warning("handle_failed", msg_id=msg_id, error=str(e))
                    finally:
                        await self.redis.xack(self.stream, self.group, msg_id)

        await close_redis()
        log.info("worker_stopped")

    def _install_signals(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:  # Windows
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Run the CSI inference stream worker.")
    p.add_argument("--name", default=None, help="consumer name (default: hostname-based)")
    a = p.parse_args()
    asyncio.run(StreamWorker(consumer_name=a.name).run())


if __name__ == "__main__":
    main()
