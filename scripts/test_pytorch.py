"""
test_pytorch.py — Smoke test using the Auto API (PyTorch backend).

Usage:
    python scripts/test_pytorch.py
    python scripts/test_pytorch.py --model ./schp-lip-20
    python scripts/test_pytorch.py --model ./schp-atr-18 --image images/image_0.jpg
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).parent.parent


def test(model_dir: str, image_path: str | None) -> None:
    print(f"Loading model from: {model_dir}")
    t0 = time.perf_counter()
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_dir, trust_remote_code=True
    )
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ── Image ────────────────────────────────────────────────────────────────
    if image_path:
        img = Image.open(image_path).convert("RGB")
        print(f"Image: {img.size[0]}×{img.size[1]}  ({Path(image_path).name})")
    else:
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        print("Image: 640×480  (random noise)")

    # ── Preprocessing ────────────────────────────────────────────────────────
    inputs = processor.preprocess(img)
    print(
        f"pixel_values: {tuple(inputs.pixel_values.shape)}  dtype={inputs.pixel_values.dtype}"
    )

    # ── Forward pass ─────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    elapsed = time.perf_counter() - t1
    print(f"Forward pass: {elapsed:.2f}s")

    # ── Output checks ────────────────────────────────────────────────────────
    size = processor.size
    H, W = size["height"], size["width"]
    assert out.logits.shape == (1, model.config.num_labels, H, W), (
        f"Unexpected logits shape: {out.logits.shape}"
    )
    print(f"logits:        {tuple(out.logits.shape)}  ✓")
    print(f"parsing_logits:{tuple(out.parsing_logits.shape)}  ✓")
    print(f"edge_logits:   {tuple(out.edge_logits.shape)}  ✓")

    # ── Segmentation map ─────────────────────────────────────────────────────
    pred = out.logits[0].argmax(dim=0).numpy()
    classes_found = sorted(np.unique(pred).tolist())
    labels = model.config.id2label
    print(f"\nClasses predicted ({len(classes_found)}):")
    for c in classes_found:
        print(f"  {c:2d}  {labels[c]}")

    print("\nAll checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for an SCHP model.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "schp-atr-18"),
        help="Path or Hub ID of the model to test (default: ./schp-atr-18)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an input image. Defaults to images/image_0.jpg if it exists, otherwise uses random noise.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_path = args.image
    if image_path is None:
        default_image = ROOT / "images" / "image_0.jpg"
        if default_image.exists():
            image_path = str(default_image)

    test(args.model, image_path)
