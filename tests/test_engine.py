"""End-to-end: train tiny -> export ONNX -> serve with PoseEstimator.

Requires torch (to build the artifact). The ONNX serving path itself uses only
onnxruntime. Module-scoped so we export the model once.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wifipose import ACTION_NAMES, NUM_FEATURES, NUM_PARTS, SEQ_LEN  # noqa: E402
from wifipose.ml.engine import PoseEstimator  # noqa: E402
from wifipose.ml.export import export_onnx  # noqa: E402
from wifipose.ml.simulator import CSISimulator  # noqa: E402
from wifipose.ml.train import make_demo_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    ckpt = make_demo_checkpoint(out_path=d / "m.pth", epochs=1, train_n=128, val_n=32, verbose=False)
    onnx = export_onnx(ckpt, out_path=d / "m.onnx", verify=True)
    return str(onnx)


def test_predict_contract(onnx_model):
    est = PoseEstimator(onnx_model)
    sample = CSISimulator(seed=5).sample(action=2)
    pred = est.predict(sample.csi)
    assert len(pred.uv_coords) == NUM_PARTS
    assert len(pred.confidence) == NUM_PARTS
    assert pred.action_name in ACTION_NAMES
    assert abs(sum(pred.action_probs.values()) - 1.0) < 1e-4
    assert pred.latency_ms >= 0


def test_validate_window_rejects_bad_shape(onnx_model):
    est = PoseEstimator(onnx_model)
    with pytest.raises(ValueError):
        est.predict(np.zeros((SEQ_LEN, NUM_FEATURES + 1), dtype=np.float32))


def test_batch_inference(onnx_model):
    est = PoseEstimator(onnx_model)
    batch = np.random.randn(8, SEQ_LEN, NUM_FEATURES).astype(np.float32)
    out = est.predict_batch(batch)
    assert out["action_logits"].shape == (8, len(ACTION_NAMES))
