"""
test_schp.py — Smoke test using the Auto API (same as from the Hub).

Usage:
    python test_schp.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).parent.parent
MODEL_DIR = str(ROOT / "schp-atr-18")
IMAGE_PATH = str(ROOT / "images" / "image_0.jpg")


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


if __name__ == "__main__":
    test(MODEL_DIR, IMAGE_PATH)
