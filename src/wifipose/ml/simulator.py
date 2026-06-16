"""Synthetic CSI signal simulator.

Real WiFi CSI capture needs specific hardware (Intel 5300 / Atheros NICs or an
ESP32 toolchain) and the original 97k-file Kaggle dataset. To make the platform
*self-contained* — runnable on any laptop or free-tier dyno with zero data
downloads — this module synthesizes physically-plausible CSI windows whose
temporal dynamics correlate with a body action.

The generator is deliberately label-consistent: the same motion statistics that
the notebook used to *derive* action labels from keypoints (mean / max joint
velocity) are what we use to *drive* the synthetic signal. So a model trained on
this simulator learns a real (if simplified) signal->action mapping, and the
live demo shows non-trivial predictions without any external dependency.

Pure NumPy: no torch import, so the streaming producer/worker stay lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wifipose import ACTION_NAMES, NUM_FEATURES, NUM_PARTS, NUM_SUBCARRIERS, SEQ_LEN

# Per-action motion profile: (base joint velocity, jitter, spike probability).
# These mirror the velocity thresholds the notebook used to label real keypoints.
_ACTION_PROFILES: dict[int, tuple[float, float, float]] = {
    0: (0.010, 0.004, 0.02),  # Normal    — moderate, steady walking
    1: (0.001, 0.001, 0.00),  # Loitering — near-stationary
    2: (0.030, 0.010, 0.05),  # Running   — high sustained velocity
    3: (0.012, 0.006, 0.30),  # Aggressive— moderate base + frequent sharp spikes
}


@dataclass(frozen=True)
class CSISample:
    csi: np.ndarray  # [T, 104] float32
    uv_coords: np.ndarray  # [24, 2] float32 — window-mean pose (model target)
    uv_seq: np.ndarray  # [T, 24, 2] float32 — full trajectory (for visualization)
    action_label: int
    action_name: str


def _base_skeleton(rng: np.random.Generator) -> np.ndarray:
    """A canonical 24-part UV skeleton in [-1, 1], randomly placed/scaled."""
    # Rough vertical layout: head high, feet low; symmetric limbs around x=0.
    ys = np.linspace(0.8, -0.8, NUM_PARTS)
    xs = 0.3 * np.sin(np.linspace(0, 3 * np.pi, NUM_PARTS))
    skel = np.stack([xs, ys], axis=1)
    skel += rng.uniform(-0.15, 0.15, size=(1, 2))  # random body position
    skel *= rng.uniform(0.7, 1.0)  # random distance/scale
    return skel.astype(np.float32)


class CSISimulator:
    """Generates labelled CSI windows for training and live streaming."""

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        num_subcarriers: int = NUM_SUBCARRIERS,
        num_features: int = NUM_FEATURES,
        seed: int | None = None,
    ):
        self.seq_len = seq_len
        self.num_subcarriers = num_subcarriers
        self.num_features = num_features
        self._rng = np.random.default_rng(seed)

    def sample(self, action: int | None = None, rng: np.random.Generator | None = None) -> CSISample:
        rng = rng or self._rng
        action = int(action if action is not None else rng.integers(0, len(ACTION_NAMES)))
        base_vel, jitter, spike_p = _ACTION_PROFILES[action]

        # --- 1. Motion trajectory (drives both CSI and ground-truth pose) ---
        skel = _base_skeleton(rng)
        uv_seq = np.empty((self.seq_len, NUM_PARTS, 2), dtype=np.float32)
        pos = skel.copy()
        # Per-joint heading; limbs (odd idx) move more than torso (even idx).
        heading = rng.uniform(-np.pi, np.pi, size=(NUM_PARTS, 1))
        limb_gain = np.where(np.arange(NUM_PARTS)[:, None] % 2 == 0, 0.6, 1.4)
        for t in range(self.seq_len):
            step = base_vel * limb_gain * np.concatenate(
                [np.cos(heading), np.sin(heading)], axis=1
            )
            step += rng.normal(0, jitter, size=pos.shape)
            if rng.random() < spike_p:  # aggressive: sudden sharp limb movement
                spike_joint = rng.integers(0, NUM_PARTS)
                step[spike_joint] += rng.normal(0, 0.08, size=2)
            pos = np.clip(pos + step, -1.0, 1.0)
            uv_seq[t] = pos
            heading += rng.normal(0, 0.2, size=heading.shape)  # wander

        # --- 2. CSI signal: subcarrier amplitudes + phases modulated by motion ---
        # Aggregate joint velocity at each timestep modulates a Doppler-like shift.
        vel = np.zeros(self.seq_len, dtype=np.float32)
        vel[1:] = np.sqrt(((np.diff(uv_seq, axis=0)) ** 2).sum(-1)).mean(-1)
        t_idx = np.arange(self.seq_len)[:, None]
        sub_idx = np.arange(self.num_subcarriers)[None, :]

        base_amp = 1.0 + 0.3 * np.sin(0.4 * sub_idx)  # static multipath fingerprint
        doppler = (vel[:, None] * 40.0) * np.sin(0.2 * sub_idx + 0.5 * t_idx)
        amp = base_amp + doppler + rng.normal(0, 0.05, size=(self.seq_len, self.num_subcarriers))

        base_phase = 0.5 * np.cos(0.3 * sub_idx)
        phase = base_phase + (vel[:, None] * 6.0) * np.cos(0.15 * sub_idx + 0.4 * t_idx)
        phase += rng.normal(0, 0.05, size=(self.seq_len, self.num_subcarriers))
        # Wrap phase to [-pi, pi] like real angle() output.
        phase = np.arctan2(np.sin(phase), np.cos(phase))

        csi = np.concatenate([amp, phase], axis=1).astype(np.float32)  # [T, 104]
        csi = self._fit_features(csi)

        return CSISample(
            csi=csi,
            uv_coords=uv_seq.mean(axis=0).astype(np.float32),
            uv_seq=uv_seq,
            action_label=action,
            action_name=ACTION_NAMES[action],
        )

    def _fit_features(self, csi: np.ndarray) -> np.ndarray:
        """Pad/trim to the canonical feature width (matches the notebook loader)."""
        f = csi.shape[1]
        if f > self.num_features:
            return csi[:, : self.num_features]
        if f < self.num_features:
            pad = np.zeros((csi.shape[0], self.num_features - f), dtype=csi.dtype)
            return np.concatenate([csi, pad], axis=1)
        return csi

    def batch(self, n: int, balanced: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (csi [n,T,104], uv [n,24,2], labels [n]) for training."""
        csis, uvs, labels = [], [], []
        for i in range(n):
            action = i % len(ACTION_NAMES) if balanced else None
            s = self.sample(action=action)
            csis.append(s.csi)
            uvs.append(s.uv_coords)
            labels.append(s.action_label)
        return (
            np.stack(csis).astype(np.float32),
            np.stack(uvs).astype(np.float32),
            np.asarray(labels, dtype=np.int64),
        )
