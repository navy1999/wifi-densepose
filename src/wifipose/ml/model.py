"""WiFi-DensePose model.

Faithfully reproduced from the original research notebook (ACVFinalProject),
refactored into a typed, importable module. The architecture is unchanged so
that checkpoints trained in the notebook load here without surprises:

    CSI window [B, T, 104]
        -> CSIEncoder        (per-timestep MLP -> embedding)
        -> DensePoseNet      (Transformer encoder -> UV coords + confidence)
        -> ActionRecognitionNet (temporal Conv1d -> action logits)

Outputs (dict):
    uv_coords     [B, 24, 2]  in [-1, 1]   (DensePose body-part UV)
    confidence    [B, 24]     in [0, 1]
    action_logits [B, 4]      (Normal / Loitering / Running / Aggressive)

Only this file imports torch. The serving path (engine.py) uses onnxruntime and
never imports torch, keeping the production image small.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wifipose import NUM_CLASSES, NUM_FEATURES, NUM_PARTS, NUM_SUBCARRIERS


class CSIEncoder(nn.Module):
    """Encodes WiFi CSI signals (amplitude + phase) into feature embeddings."""

    def __init__(self, num_subcarriers: int = NUM_SUBCARRIERS, embedding_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_subcarriers * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, 104] -> [B, T, E]
        return self.encoder(x)


class DensePoseNet(nn.Module):
    """Maps CSI feature sequence to DensePose UV coordinates for 24 body parts."""

    def __init__(self, embedding_dim: int = 256, num_parts: int = NUM_PARTS):
        super().__init__()
        self.num_parts = num_parts

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.uv_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_parts * 2),
        )
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_parts),
            nn.Sigmoid(),
        )

    def forward(self, csi_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.transformer(csi_features)
        last = features[:, -1, :]  # use final timestep as the pose summary
        uv = torch.tanh(self.uv_predictor(last)).view(-1, self.num_parts, 2)
        confidence = self.confidence_predictor(last)
        return uv, confidence


class ActionRecognitionNet(nn.Module):
    """Temporal model for security action classification over the CSI window."""

    def __init__(self, embedding_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embedding_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, csi_features: torch.Tensor) -> torch.Tensor:
        x = csi_features.permute(0, 2, 1)  # [B, T, E] -> [B, E, T]
        x = self.temporal_conv(x).squeeze(-1)
        return self.classifier(x)


class WiFiDensePoseSystem(nn.Module):
    """Complete end-to-end system: CSI -> pose + action."""

    def __init__(
        self,
        num_subcarriers: int = NUM_SUBCARRIERS,
        embedding_dim: int = 256,
        num_parts: int = NUM_PARTS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.csi_encoder = CSIEncoder(num_subcarriers, embedding_dim)
        self.densepose_net = DensePoseNet(embedding_dim, num_parts)
        self.action_net = ActionRecognitionNet(embedding_dim, num_classes)

    def forward(self, csi_data: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.csi_encoder(csi_data)
        uv, confidence = self.densepose_net(feats)
        action_logits = self.action_net(feats)
        return {
            "uv_coords": uv,
            "confidence": confidence,
            "action_logits": action_logits,
        }


def build_model() -> WiFiDensePoseSystem:
    """Construct the model with the platform's canonical I/O contract."""
    assert NUM_FEATURES == NUM_SUBCARRIERS * 2
    return WiFiDensePoseSystem(
        num_subcarriers=NUM_SUBCARRIERS,
        embedding_dim=256,
        num_parts=NUM_PARTS,
        num_classes=NUM_CLASSES,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
