"""The simulator must produce correctly-shaped, label-consistent data."""

import numpy as np

from wifipose import ACTION_NAMES, NUM_FEATURES, NUM_PARTS, SEQ_LEN
from wifipose.ml.simulator import CSISimulator


def test_sample_shapes():
    sim = CSISimulator(seed=1)
    s = sim.sample(action=2)
    assert s.csi.shape == (SEQ_LEN, NUM_FEATURES)
    assert s.uv_coords.shape == (NUM_PARTS, 2)
    assert s.uv_seq.shape == (SEQ_LEN, NUM_PARTS, 2)
    assert s.action_label == 2
    assert s.action_name == "Running"
    assert s.csi.dtype == np.float32


def test_uv_in_range():
    sim = CSISimulator(seed=2)
    for a in range(len(ACTION_NAMES)):
        s = sim.sample(action=a)
        assert s.uv_coords.min() >= -1.0 and s.uv_coords.max() <= 1.0


def test_running_moves_more_than_loitering():
    """Motion statistics must respect the action profiles (this is what makes
    the synthetic labels learnable)."""
    sim = CSISimulator(seed=3)
    def avg_velocity(action):
        vels = []
        for _ in range(20):
            s = sim.sample(action=action)
            vels.append(np.sqrt((np.diff(s.uv_seq, axis=0) ** 2).sum(-1)).mean())
        return np.mean(vels)

    assert avg_velocity(2) > avg_velocity(0) > avg_velocity(1)  # Running > Normal > Loitering


def test_batch_is_balanced():
    sim = CSISimulator(seed=4)
    X, U, y = sim.batch(8)
    assert X.shape == (8, SEQ_LEN, NUM_FEATURES)
    assert U.shape == (8, NUM_PARTS, 2)
    assert sorted(y.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]
