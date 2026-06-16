"""Model architecture contract (requires the ml extras / torch)."""

import pytest

torch = pytest.importorskip("torch")

from wifipose import NUM_CLASSES, NUM_FEATURES, NUM_PARTS, SEQ_LEN  # noqa: E402
from wifipose.ml.model import build_model, count_parameters  # noqa: E402


def test_forward_shapes():
    model = build_model().eval()
    x = torch.randn(3, SEQ_LEN, NUM_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out["uv_coords"].shape == (3, NUM_PARTS, 2)
    assert out["confidence"].shape == (3, NUM_PARTS)
    assert out["action_logits"].shape == (3, NUM_CLASSES)


def test_output_ranges():
    model = build_model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, SEQ_LEN, NUM_FEATURES))
    assert out["uv_coords"].min() >= -1 and out["uv_coords"].max() <= 1  # tanh
    assert out["confidence"].min() >= 0 and out["confidence"].max() <= 1  # sigmoid


def test_param_count_reasonable():
    n = count_parameters(build_model())
    assert 1_000_000 < n < 10_000_000  # ~3.3M expected
