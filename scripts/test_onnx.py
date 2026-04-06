"""
test_onnx.py — Test the exported SCHP ONNX model on a real image.

Usage:
    python test_onnx.py
    python test_onnx.py --image images/image_0.jpg --model onnx/schp-atr-18.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from schp.image_processing_schp import SCHPImageProcessor

ROOT = Path(__file__).parent.parent
ONNX_MODEL = str(ROOT / "onnx" / "schp-atr-18.onnx")
IMAGE_PATH = str(ROOT / "images" / "image_0.jpg")
ORT_THREADS = 8


def _make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


_ATR_LABELS = [
    "Background",
    "Hat",
    "Hair",
    "Sunglasses",
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Left-shoe",
    "Right-shoe",
    "Face",
    "Left-leg",
    "Right-leg",
    "Left-arm",
    "Right-arm",
    "Bag",
    "Scarf",
]

_ATR_COLORS = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [255, 0, 0],
        [0, 85, 0],
        [170, 0, 51],
        [255, 85, 0],
        [0, 0, 85],
        [0, 119, 221],
        [85, 85, 0],
        [0, 85, 85],
        [85, 51, 0],
        [52, 86, 128],
        [0, 128, 0],
        [0, 0, 255],
        [51, 170, 221],
        [0, 255, 255],
        [85, 255, 170],
        [170, 255, 85],
    ],
    dtype=np.uint8,
)


def run(image_path: str, model_path: str) -> None:
    # ── Load & preprocess ────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    print(f"Input image: {image.size} ({image_path})")

    processor = SCHPImageProcessor()
    inputs = processor(images=image, return_tensors="np")
    pixel_values: np.ndarray = inputs["pixel_values"]  # (1, 3, 512, 512) float32

    # ── ONNX inference ───────────────────────────────────────────────────────
    sess = _make_session(model_path)
    outputs = sess.run(None, {"pixel_values": pixel_values})

    logits = outputs[0]  # (1, 18, 512, 512)
    print(f"Logits shape: {logits.shape}")

    # ── Argmax → label map ───────────────────────────────────────────────────
    label_map = np.argmax(logits[0], axis=0)  # (512, 512)

    # ── Color map ────────────────────────────────────────────────────────────
    color_map = _ATR_COLORS[label_map]  # (512, 512, 3)
    color_image = Image.fromarray(color_map).resize(image.size, Image.NEAREST)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    out_path = str(out_dir / f"{stem}_onnx_seg.png")
    color_image.save(out_path)
    print(f"Segmentation map saved to: {out_path}")

    # ── Print detected classes ───────────────────────────────────────────────
    unique_labels = np.unique(label_map)
    print("\nDetected classes:")
    for lbl in unique_labels:
        if lbl > 0:  # skip background
            print(f"  [{lbl:2d}] {_ATR_LABELS[lbl]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--model", default=ONNX_MODEL)
    args = parser.parse_args()
    run(args.image, args.model)
