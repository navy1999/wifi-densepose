"""Training entry points.

Two paths share the same model and loss design (faithful to the notebook):

1. `make_demo_checkpoint` / `wifipose-checkpoint` — trains on the synthetic
   simulator so anyone can produce a working checkpoint in ~30s on CPU, with NO
   dataset download. This is what powers the live demo and CI.

2. `train_on_kaggle` / `wifipose-train` — the real-data path that mirrors the
   notebook (h5py CSI + .npy keypoints, ConcatDataset of both Kaggle sources).
   Requires the optional `ml` extras and the downloaded datasets.

Loss: L1 on UV coords (lambda 0.5) + CrossEntropy on action logits (lambda 1.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wifipose import SEQ_LEN
from wifipose.ml.simulator import CSISimulator


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Training needs the 'ml' extras. Install with:  pip install -e \".[ml]\""
        ) from e
    import torch

    return torch


def make_demo_checkpoint(
    out_path: str | Path = "checkpoints/wifipose.pth",
    epochs: int = 8,
    train_n: int = 1024,
    val_n: int = 256,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> Path:
    """Train the model briefly on simulated CSI and save a state_dict."""
    torch = _require_torch()
    import torch.nn as nn

    from wifipose.ml.model import build_model, count_parameters

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = CSISimulator(seed=seed)
    Xtr, Utr, Ytr = sim.batch(train_n)
    Xva, Uva, Yva = sim.batch(val_n)
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    Xtr, Utr, Ytr = to(Xtr), to(Utr), to(Ytr)
    Xva, Uva, Yva = to(Xva), to(Uva), to(Yva)

    model = build_model().to(device)
    pose_crit = nn.L1Loss()
    act_crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lambda_pose, lambda_action = 0.5, 1.0

    if verbose:
        print(f"Model params: {count_parameters(model):,} | device: {device}")

    n = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = lambda_pose * pose_crit(out["uv_coords"], Utr[idx]) + lambda_action * act_crit(
                out["action_logits"], Ytr[idx]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(Xva)
            vloss = (
                lambda_pose * pose_crit(out["uv_coords"], Uva)
                + lambda_action * act_crit(out["action_logits"], Yva)
            ).item()
            acc = (out["action_logits"].argmax(1) == Yva).float().mean().item()
        if verbose:
            print(f"epoch {epoch:>2}/{epochs}  val_loss={vloss:.4f}  val_acc={acc*100:5.1f}%")
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    if verbose:
        print(f"Saved checkpoint -> {out_path} (best val_loss={best_val:.4f})")
    return out_path


def train_on_kaggle(
    data_roots: list[str],
    out_path: str | Path = "checkpoints/wifipose.pth",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Path:
    """Real-data training mirroring the notebook (requires downloaded datasets)."""
    torch = _require_torch()
    import torch.nn as nn
    from torch.utils.data import ConcatDataset, DataLoader

    from wifipose.ml.dataset import KaggleWiFiPoseDataset
    from wifipose.ml.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_sets = [KaggleWiFiPoseDataset(r, split="train_data", seq_len=SEQ_LEN) for r in data_roots]
    test_sets = [KaggleWiFiPoseDataset(r, split="test_data", seq_len=SEQ_LEN) for r in data_roots]
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size)

    model = build_model().to(device)
    pose_crit, act_crit = nn.L1Loss(), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    lambda_pose, lambda_action = 0.5, 1.0
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            csi = batch["csi"].to(device)
            uv = batch["uv_coords"].to(device)
            y = batch["action_label"].to(device)
            opt.zero_grad()
            out = model(csi)
            loss = lambda_pose * pose_crit(out["uv_coords"], uv) + lambda_action * act_crit(
                out["action_logits"], y
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vtot, vn, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                csi = batch["csi"].to(device)
                uv = batch["uv_coords"].to(device)
                y = batch["action_label"].to(device)
                out = model(csi)
                loss = lambda_pose * pose_crit(out["uv_coords"], uv) + lambda_action * act_crit(
                    out["action_logits"], y
                )
                vtot += loss.item() * csi.size(0)
                vn += csi.size(0)
                correct += (out["action_logits"].argmax(1) == y).sum().item()
        vloss = vtot / max(vn, 1)
        sched.step(vloss)
        print(f"epoch {epoch}/{epochs}  val_loss={vloss:.4f}  val_acc={100*correct/max(vn,1):.1f}%")
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"Saved checkpoint -> {out_path}")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────
def make_demo_checkpoint_cli() -> None:
    p = argparse.ArgumentParser(description="Train a demo checkpoint on simulated CSI.")
    p.add_argument("--out", default="checkpoints/wifipose.pth")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make_demo_checkpoint(out_path=a.out, epochs=a.epochs, seed=a.seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Train WiFi-DensePose on real Kaggle data.")
    p.add_argument("--data-root", action="append", required=True, help="repeatable dataset root")
    p.add_argument("--out", default="checkpoints/wifipose.pth")
    p.add_argument("--epochs", type=int, default=25)
    a = p.parse_args()
    train_on_kaggle(a.data_root, out_path=a.out, epochs=a.epochs)


if __name__ == "__main__":
    main()
