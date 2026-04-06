"""
download_checkpoints.py — Download original SCHP pretrained checkpoints from Google Drive.

Usage:
    python scripts/download_checkpoints.py --dataset atr
    python scripts/download_checkpoints.py --dataset lip
    python scripts/download_checkpoints.py --dataset all
"""

import argparse
from pathlib import Path

import gdown

ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"

_CHECKPOINTS = {
    "atr": {
        "id": "1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP",
        "filename": "exp-schp-201908301523-atr.pth",
    },
    "lip": {
        "id": "1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
        "filename": "exp-schp-201908261155-lip.pth",
    },
    "pascal": {
        "id": "1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE",
        "filename": "exp-schp-201908270938-pascal-person-part.pth",
    },
}


def download(dataset: str) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = list(_CHECKPOINTS.keys()) if dataset == "all" else [dataset]

    for name in targets:
        info = _CHECKPOINTS[name]
        output = CHECKPOINTS_DIR / info["filename"]
        if output.exists():
            print(f"[{name}] Already exists: {output}")
            continue
        print(f"[{name}] Downloading {info['filename']}...")
        gdown.download(id=info["id"], output=str(output), quiet=False)
        print(f"[{name}] Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SCHP pretrained checkpoints."
    )
    parser.add_argument(
        "--dataset",
        choices=["atr", "lip", "pascal", "all"],
        default="lip",
        help="Which checkpoint to download (default: lip).",
    )
    args = parser.parse_args()
    download(args.dataset)
