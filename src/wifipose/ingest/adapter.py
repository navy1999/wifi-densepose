"""CSI normalization adapter — the bridge between messy real hardware and the
model's fixed input contract.

Different capture sources emit CSI in different shapes:
  - ESP32 (esp-csi): per packet, interleaved int8 [imag0, real0, imag1, real1, ...]
    for N subcarriers (N = 64 for HT20, more for HT40 / 11ac).
  - Nexmon / Intel 5300: complex per subcarrier, different N.
  - Pre-extracted amplitude or amplitude+phase arrays.

The model always wants a window of shape [SEQ_LEN, 104], where 104 =
52 subcarriers x (amplitude, phase) laid out as [amp(52) | phase(52)] per
timestep — exactly how the training notebook built features. This module maps
any of the above into that contract (pad/trim subcarriers and time), so the
platform genuinely "takes whatever CSI you send and classifies it."

Pure NumPy: shared by the server ingestion path and the edge capture agent.
"""

from __future__ import annotations

import re

import numpy as np

from wifipose import NUM_FEATURES, NUM_SUBCARRIERS, SEQ_LEN

# Supported wire formats for a single CSI packet.
FORMAT_COMPLEX_INTERLEAVED = "complex_interleaved"  # [im0, re0, im1, re1, ...] (ESP32 raw)
FORMAT_COMPLEX_PAIRS = "complex_pairs"  # [[re0, im0], [re1, im1], ...]
FORMAT_AMPLITUDE = "amplitude"  # [a0, a1, ...] (phase unknown -> zeros)
FORMAT_AMP_PHASE = "amp_phase"  # already [amp..., phase...] feature vector
SUPPORTED_FORMATS = (
    FORMAT_COMPLEX_INTERLEAVED,
    FORMAT_COMPLEX_PAIRS,
    FORMAT_AMPLITUDE,
    FORMAT_AMP_PHASE,
)


class CSIFormatError(ValueError):
    """Raised when a packet cannot be parsed into the expected format."""


def _pad_trim(vec: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or trim a 1-D array to exactly length n."""
    if vec.shape[0] == n:
        return vec
    if vec.shape[0] > n:
        return vec[:n]
    return np.concatenate([vec, np.zeros(n - vec.shape[0], dtype=vec.dtype)])


def _complex_to_features(csi: np.ndarray) -> np.ndarray:
    """Complex CSI per subcarrier -> [amp(52) | phase(52)] = 104 float32."""
    amp = _pad_trim(np.abs(csi).astype(np.float32), NUM_SUBCARRIERS)
    phase = np.angle(csi).astype(np.float32)
    phase = np.arctan2(np.sin(phase), np.cos(phase))  # wrap to [-pi, pi]
    phase = _pad_trim(phase, NUM_SUBCARRIERS)
    return np.concatenate([amp, phase]).astype(np.float32)


def normalize_packet(packet, fmt: str = FORMAT_COMPLEX_INTERLEAVED) -> np.ndarray:
    """Normalize one CSI packet (any supported format) to a [104] feature vector."""
    arr = np.asarray(packet, dtype=np.float32).ravel()
    if arr.size == 0:
        raise CSIFormatError("Empty CSI packet.")

    if fmt == FORMAT_COMPLEX_INTERLEAVED:
        if arr.size % 2 != 0:
            arr = arr[:-1]  # drop a trailing odd sample rather than fail
        imag = arr[0::2]
        real = arr[1::2]
        return _complex_to_features(real + 1j * imag)

    if fmt == FORMAT_COMPLEX_PAIRS:
        pairs = np.asarray(packet, dtype=np.float32).reshape(-1, 2)
        return _complex_to_features(pairs[:, 0] + 1j * pairs[:, 1])

    if fmt == FORMAT_AMPLITUDE:
        amp = _pad_trim(arr, NUM_SUBCARRIERS)
        return np.concatenate([amp, np.zeros(NUM_SUBCARRIERS, dtype=np.float32)])

    if fmt == FORMAT_AMP_PHASE:
        return _pad_trim(arr, NUM_FEATURES)

    raise CSIFormatError(f"Unsupported CSI format '{fmt}'. Use one of {SUPPORTED_FORMATS}.")


def build_window(packets, fmt: str = FORMAT_COMPLEX_INTERLEAVED, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Normalize a list of packets into a [seq_len, 104] window (pad/trim in time)."""
    if not packets:
        raise CSIFormatError("No packets provided.")
    feats = np.stack([normalize_packet(p, fmt) for p in packets]).astype(np.float32)

    t = feats.shape[0]
    if t == seq_len:
        return feats
    if t > seq_len:
        return feats[-seq_len:]  # most recent window
    pad = np.repeat(feats[-1:], seq_len - t, axis=0)  # edge-pad to length
    return np.concatenate([feats, pad], axis=0).astype(np.float32)


# ── ESP32 (esp-csi) serial line parser ───────────────────────────────────────
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


def parse_esp32_line(line: str) -> list[float] | None:
    """Parse one esp-csi `CSI_DATA` serial line into interleaved [im, re, ...] ints.

    esp-csi prints CSV lines ending in a bracketed array of signed ints, e.g.
    `CSI_DATA,STA,...,[12 -3 8 5 ...]`. Returns the interleaved list (to be fed
    with FORMAT_COMPLEX_INTERLEAVED), or None for non-CSI lines.
    """
    if "CSI_DATA" not in line:
        return None
    m = _BRACKET_RE.search(line)
    if not m:
        return None
    body = m.group(1).strip().replace(",", " ")
    try:
        vals = [float(tok) for tok in body.split() if tok]
    except ValueError:
        return None
    return vals or None
