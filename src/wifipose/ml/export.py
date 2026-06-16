"""Export a trained PyTorch checkpoint to ONNX for production serving.

The exported graph uses a dynamic batch axis (and dynamic sequence length) so a
single artifact serves both single-frame requests and batched throughput. We
verify numerical parity (torch vs onnxruntime) before declaring success.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wifipose import NUM_FEATURES, SEQ_LEN


def export_onnx(
    checkpoint: str | Path,
    out_path: str | Path = "checkpoints/wifipose.onnx",
    opset: int = 17,
    verify: bool = True,
) -> Path:
    import torch

    from wifipose.ml.model import build_model, count_parameters

    checkpoint, out_path = Path(checkpoint), Path(out_path)
    model = build_model()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {checkpoint} ({count_parameters(model):,} params)")

    dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The model returns a dict; ONNX export needs ordered tuple outputs.
    class _Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(x)
            return o["uv_coords"], o["confidence"], o["action_logits"]

    export_kwargs = {
        "input_names": ["csi"],
        "output_names": ["uv_coords", "confidence", "action_logits"],
        "dynamic_axes": {
            "csi": {0: "batch", 1: "seq"},
            "uv_coords": {0: "batch"},
            "confidence": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }
    try:
        # Force the stable TorchScript exporter (deterministic dynamic_axes
        # support, no onnxscript runtime dependency). Newer torch defaults to the
        # dynamo exporter; `dynamo=False` keeps behaviour predictable.
        torch.onnx.export(_Wrapped(model), dummy, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch (<2.x) has no `dynamo` kwarg.
        torch.onnx.export(_Wrapped(model), dummy, str(out_path), **export_kwargs)
    print(f"Exported ONNX -> {out_path}")

    if verify:
        _verify_parity(model, out_path)
    return out_path


def _verify_parity(torch_model, onnx_path: Path, atol: float = 1e-3) -> None:
    import onnxruntime as ort
    import torch

    # torch.onnx.export can leave the module in training mode; force eval so the
    # reference forward has dropout disabled (matches the exported graph).
    torch_model.eval()
    x = torch.randn(4, SEQ_LEN, NUM_FEATURES)
    with torch.no_grad():
        ref = torch_model(x)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"csi": x.numpy()})
    names = ["uv_coords", "confidence", "action_logits"]
    for name, got in zip(names, outs, strict=True):
        exp = ref[name].numpy()
        max_diff = float(np.abs(exp - got).max())
        status = "OK" if max_diff < atol else "MISMATCH"
        print(f"  parity[{name}]: max_abs_diff={max_diff:.2e}  {status}")
        if max_diff >= atol:
            raise RuntimeError(f"ONNX parity check failed for {name} (max diff {max_diff:.2e}).")
    print("ONNX numerical parity verified.")


def main() -> None:
    p = argparse.ArgumentParser(description="Export checkpoint to ONNX.")
    p.add_argument("--checkpoint", default="checkpoints/wifipose.pth")
    p.add_argument("--out", default="checkpoints/wifipose.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    export_onnx(a.checkpoint, a.out, opset=a.opset, verify=not a.no_verify)


if __name__ == "__main__":
    main()
