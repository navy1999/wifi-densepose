"""CSI normalization adapter — heterogeneous formats -> canonical [30,104]."""

import numpy as np
import pytest

from wifipose import NUM_FEATURES, NUM_SUBCARRIERS, SEQ_LEN
from wifipose.ingest.adapter import (
    FORMAT_AMP_PHASE,
    FORMAT_AMPLITUDE,
    FORMAT_COMPLEX_INTERLEAVED,
    FORMAT_COMPLEX_PAIRS,
    CSIFormatError,
    build_window,
    normalize_packet,
    parse_esp32_line,
)


def test_complex_interleaved_to_104():
    # 64 subcarriers -> 128 interleaved ints -> trimmed to 104 features
    packet = list(range(128))
    feats = normalize_packet(packet, FORMAT_COMPLEX_INTERLEAVED)
    assert feats.shape == (NUM_FEATURES,)
    assert feats.dtype == np.float32


def test_complex_pairs_matches_interleaved():
    interleaved = [3, 4, 0, 1]  # (im=3,re=4), (im=0,re=1)
    pairs = [[4, 3], [1, 0]]  # (re,im)
    a = normalize_packet(interleaved, FORMAT_COMPLEX_INTERLEAVED)
    b = normalize_packet(pairs, FORMAT_COMPLEX_PAIRS)
    # amplitude of first subcarrier = |4+3j| = 5
    assert abs(a[0] - 5.0) < 1e-5
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_amplitude_format_zero_phase():
    feats = normalize_packet([1.0] * 52, FORMAT_AMPLITUDE)
    assert feats.shape == (NUM_FEATURES,)
    assert np.allclose(feats[NUM_SUBCARRIERS:], 0.0)  # phase half is zeros


def test_amp_phase_padding_and_trim():
    assert normalize_packet([1.0] * 50, FORMAT_AMP_PHASE).shape == (NUM_FEATURES,)
    assert normalize_packet([1.0] * 200, FORMAT_AMP_PHASE).shape == (NUM_FEATURES,)


def test_phase_wrapped_to_pi():
    feats = normalize_packet([1000, 1, 1000, 1], FORMAT_COMPLEX_INTERLEAVED)
    phase = feats[NUM_SUBCARRIERS:]
    assert phase.min() >= -np.pi - 1e-5 and phase.max() <= np.pi + 1e-5


def test_build_window_pads_and_trims_time():
    short = build_window([[1.0] * 104] * 10, FORMAT_AMP_PHASE, seq_len=SEQ_LEN)
    long = build_window([[1.0] * 104] * 50, FORMAT_AMP_PHASE, seq_len=SEQ_LEN)
    assert short.shape == (SEQ_LEN, NUM_FEATURES)
    assert long.shape == (SEQ_LEN, NUM_FEATURES)


def test_empty_raises():
    with pytest.raises(CSIFormatError):
        normalize_packet([], FORMAT_AMP_PHASE)
    with pytest.raises(CSIFormatError):
        build_window([], FORMAT_AMP_PHASE)


def test_unknown_format_raises():
    with pytest.raises(CSIFormatError):
        normalize_packet([1, 2, 3], "nonsense")


@pytest.mark.parametrize(
    "line,expect",
    [
        ("CSI_DATA,STA,aa:bb,-40,11,[1 2 3 4]", [1.0, 2.0, 3.0, 4.0]),
        ("CSI_DATA,...,[1,2,3,4]", [1.0, 2.0, 3.0, 4.0]),
        ("some boot log line", None),
        ("CSI_DATA but no bracket", None),
    ],
)
def test_parse_esp32_line(line, expect):
    assert parse_esp32_line(line) == expect
