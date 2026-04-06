"""
export_onnx.py — Export the converted SCHP model to ONNX.

Usage:
    python export_onnx.py
    python export_onnx.py --model ./schp-atr-18 --output schp-atr-18.onnx

Requirements:
    pip install onnx onnxruntime
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSemanticSegmentation

ROOT = Path(__file__).parent.parent
MODEL_DIR = str(ROOT / "schp-atr-18")
OUTPUT_PATH = str(ROOT / "schp-atr-18" / "onnx" / "schp-atr-18.onnx")


def export(model_dir: str, output_path: str) -> None:
    print(f"Loading model from: {model_dir}")
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_dir, trust_remote_code=True
    )
    model.eval()

    input_size = model.config.input_size
    dummy = torch.zeros(1, 3, input_size, input_size)
    print(f"Input size: {input_size}×{input_size}")

    print(f"Exporting to: {output_path}")
    batch = torch.export.Dim("batch", min=1, max=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args=(),
        kwargs={"pixel_values": dummy, "return_dict": False},
        f=output_path,
        opset_version=21,
        input_names=["pixel_values"],
        output_names=["logits", "parsing_logits", "edge_logits"],
        dynamic_shapes={"pixel_values": {0: batch}, "return_dict": None},
    )
    print("Export done.")

    # ── Optional: verify with onnxruntime ────────────────────────────────────
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"pixel_values": dummy.numpy()})
        print(f"ORT verification OK — logits shape: {out[0].shape}")

        # Check outputs match PyTorch
        with torch.no_grad():
            pt_out = model(pixel_values=dummy)
        np.testing.assert_allclose(out[0], pt_out.logits.numpy(), rtol=1e-3, atol=1e-4)
        print("PyTorch ↔ ONNX outputs match  ✓")
    except ImportError:
        print("onnxruntime not installed — skipping verification.")
        print("Install with: pip install onnxruntime")

    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"\nONNX model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SCHP to ONNX.")
    parser.add_argument("--model", default=MODEL_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    export(args.model, args.output)
