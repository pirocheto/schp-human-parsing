"""
download_images.py — Download sample images from the Fashionpedia dataset.

Usage:
    python scripts/download_images.py
    python scripts/download_images.py --n 20 --output images --name photo
"""

import argparse
from pathlib import Path

from datasets import load_dataset

N = 10
OUTPUT_DIR = Path("images")
SEED = 42
NAME = "image"


def download_images(n: int, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "detection-datasets/fashionpedia", split="val", streaming=True
    ).shuffle(seed=SEED, buffer_size=1000)

    for i, sample in enumerate(dataset.take(n)):
        img = sample["image"]
        path = output_dir / f"{name}_{i}.jpg"
        img.save(path)
        print(f"Image {i} saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample images.")
    parser.add_argument(
        "--n", type=int, default=N, help="Number of images to download (default: 10)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: images).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=NAME,
        help="Base filename for images (default: image).",
    )
    args = parser.parse_args()
    download_images(args.n, args.output, args.name)
