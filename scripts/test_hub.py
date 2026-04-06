"""
test_hub.py — Test the model loaded directly from the HuggingFace Hub.

Usage:
    python scripts/test_hub.py
    python scripts/test_hub.py --image path/to/image.jpg
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

REPO_ID = "pirocheto/schp-atr-18"


def test(image_path: str | None) -> None:
    print(f"Loading model from Hub: {REPO_ID}")
    t0 = time.perf_counter()
    model = AutoModelForSemanticSegmentation.from_pretrained(
        REPO_ID, trust_remote_code=True
    )
    processor = AutoImageProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ── Image ─────────────────────────────────────────────────────────────────
    if image_path:
        img = Image.open(image_path).convert("RGB")
        print(f"Image: {img.size[0]}×{img.size[1]}  ({Path(image_path).name})")
    else:
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        print("Image: 640×480  (random noise — pass --image for a real result)")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    inputs = processor(img, return_tensors="pt")
    print(
        f"pixel_values: {tuple(inputs.pixel_values.shape)}  dtype={inputs.pixel_values.dtype}"
    )

    # ── Forward pass ──────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    print(f"Forward pass: {time.perf_counter() - t1:.2f}s")

    # ── Output checks ─────────────────────────────────────────────────────────
    H = processor.size["height"]
    W = processor.size["width"]
    assert out.logits.shape == (1, model.config.num_labels, H, W), (
        f"Unexpected logits shape: {out.logits.shape}"
    )
    print(f"logits:        {tuple(out.logits.shape)}  ✓")

    # ── Segmentation map ──────────────────────────────────────────────────────
    pred = out.logits[0].argmax(dim=0).numpy()
    classes_found = sorted(np.unique(pred).tolist())
    labels = model.config.id2label
    print(f"\nClasses predicted ({len(classes_found)}):")
    for c in classes_found:
        print(f"  {c:2d}  {labels[c]}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to an input image")
    args = parser.parse_args()
    test(args.image)
