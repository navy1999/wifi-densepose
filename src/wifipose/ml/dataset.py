"""Real-data loader for the Kaggle WiFi-pose datasets (notebook-faithful).

Used only by the optional real-data training path. Requires the `ml` extras
(h5py). Action labels are inferred from ground-truth keypoint motion exactly as
in the original notebook, so the two Kaggle sources can be merged with
`torch.utils.data.ConcatDataset`.

Supported datasets (identical schema, verified in the notebook):
    - nguyenngonhatnam/wifipose-dataset
    - nghinguyenvinh/csi-human-in-wifi  (.../Person-in-WiFi-3D)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wifipose import ACTION_NAMES, NUM_FEATURES, NUM_PARTS, SEQ_LEN

try:
    import h5py
    from torch.utils.data import Dataset

    _BASE = Dataset
except ImportError:  # pragma: no cover - allows import without ml extras
    _BASE = object  # type: ignore
    h5py = None  # type: ignore


class KaggleWiFiPoseDataset(_BASE):  # type: ignore[misc]
    """Loads CSI (.mat via h5py) + keypoints (.npy); infers action from motion."""

    def __init__(self, data_dir: str | Path, split: str = "train_data", seq_len: int = SEQ_LEN):
        if h5py is None:  # pragma: no cover
            raise ImportError('Real-data loading needs the ml extras: pip install -e ".[ml]"')
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.action_names = list(ACTION_NAMES)

        split_dir = self.data_dir / split
        self.csi_dir = split_dir / "csi"
        self.kpt_dir = split_dir / "keypoint"
        list_file = split_dir / f"{split}_list.txt"
        if not self.csi_dir.exists() or not self.kpt_dir.exists():
            raise FileNotFoundError(f"Expected 'csi' and 'keypoint' under {split_dir}")
        if not list_file.exists():
            raise FileNotFoundError(f"List file not found: {list_file}")

        with open(list_file) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        self.ids = [
            sid
            for sid in ids
            if (self.csi_dir / f"{sid}.mat").exists() and (self.kpt_dir / f"{sid}.npy").exists()
        ]
        if not self.ids:
            raise RuntimeError(f"No paired samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_csi(self, sid: str) -> np.ndarray:
        with h5py.File(self.csi_dir / f"{sid}.mat", "r") as f:
            dset = f["csi_serial"] if "csi_serial" in f else f[list(f.keys())[0]]
            csi = np.array(dset)
        if csi.dtype.fields and "real" in csi.dtype.fields and "imag" in csi.dtype.fields:
            c = csi["real"] + 1j * csi["imag"]
            csi = np.concatenate([np.abs(c), np.angle(c)], axis=-1)
        if csi.ndim >= 3:
            csi = csi.reshape(csi.shape[0], -1)
        if csi.shape[1] > NUM_FEATURES:
            csi = csi[:, :NUM_FEATURES]
        elif csi.shape[1] < NUM_FEATURES:
            pad = np.zeros((csi.shape[0], NUM_FEATURES - csi.shape[1]), dtype=csi.dtype)
            csi = np.concatenate([csi, pad], axis=1)
        return csi.astype(np.float32)

    def _load_keypoint(self, sid: str) -> np.ndarray:
        kp = np.load(self.kpt_dir / f"{sid}.npy")
        if kp.ndim == 4:
            kp = kp[:, 0, :, :]
        uv = kp[..., :2]
        m = np.abs(uv).max()
        if m > 0:
            uv = uv / (m * 2)
        parts = uv.shape[1]
        if parts < NUM_PARTS:
            pad = np.zeros((uv.shape[0], NUM_PARTS - parts, 2), dtype=uv.dtype)
            uv = np.concatenate([uv, pad], axis=1)
        elif parts > NUM_PARTS:
            uv = uv[:, :NUM_PARTS, :]
        return np.clip(uv, -1, 1).astype(np.float32)

    @staticmethod
    def infer_action_label(uv_seq: np.ndarray) -> int:
        """Notebook heuristic: derive action from keypoint velocity statistics."""
        if uv_seq.shape[0] < 2:
            return 0
        motion = np.sqrt((np.diff(uv_seq, axis=0) ** 2).sum(-1))
        avg_v, max_v = motion.mean(), motion.max()
        if avg_v > 0.02:
            return 2  # Running
        if avg_v < 0.002:
            return 1  # Loitering
        if max_v > 0.05:
            return 3  # Aggressive
        return 0  # Normal

    def __getitem__(self, idx: int) -> dict:
        import torch

        sid = self.ids[idx]
        csi = self._load_csi(sid)
        uv = self._load_keypoint(sid)

        t = csi.shape[0]
        if t >= self.seq_len:
            start = np.random.randint(0, t - self.seq_len + 1)
            csi_w = csi[start : start + self.seq_len]
            uv_w = uv[start : start + self.seq_len]
        else:
            csi_w = np.pad(csi, ((0, self.seq_len - t), (0, 0)), mode="edge")
            uv_w = np.pad(uv, ((0, self.seq_len - t), (0, 0), (0, 0)), mode="edge")

        return {
            "csi": torch.from_numpy(csi_w.astype(np.float32)),
            "uv_coords": torch.from_numpy(uv_w.mean(axis=0).astype(np.float32)),
            "action_label": int(self.infer_action_label(uv_w)),
        }
