"""WiFi Pose Intelligence Platform.

Turn raw WiFi Channel State Information (CSI) into human pose + action
intelligence — no cameras. Production ML serving (ONNX), time-series storage
(TimescaleDB/Postgres + pgvector), streaming ingest (Redis), and an LLM
natural-language analytics layer.
"""

__version__ = "0.1.0"

# Canonical model I/O contract — imported everywhere so the shape is defined once.
SEQ_LEN = 30  # timesteps per CSI window
NUM_FEATURES = 104  # 52 subcarriers x (amplitude + phase)
NUM_SUBCARRIERS = 52
NUM_PARTS = 24  # DensePose body parts (UV pairs)
NUM_CLASSES = 4
ACTION_NAMES = ("Normal", "Loitering", "Running", "Aggressive")
