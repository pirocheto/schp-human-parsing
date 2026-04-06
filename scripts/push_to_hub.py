"""
push_to_hub.py — Push SCHP model + ONNX files to Hugging Face Hub.

Usage:
    huggingface-cli login   # only once
    python scripts/push_to_hub.py --dataset atr
    python scripts/push_to_hub.py --dataset lip
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).parent.parent

_CONFIGS = {
    "atr": {
        "repo_id": "pirocheto/schp-atr-18",
        "model_dir": ROOT / "schp-atr-18",
        "onnx_files": [
            "schp-atr-18.onnx",
            "schp-atr-18.onnx.data",
            "schp-atr-18-int8-static.onnx",
        ],
    },
    "lip": {
        "repo_id": "pirocheto/schp-lip-20",
        "model_dir": ROOT / "schp-lip-20",
        "onnx_files": [
            "schp-lip-20.onnx",
            "schp-lip-20.onnx.data",
            "schp-lip-20-int8-static.onnx",
        ],
    },
    "pascal": {
        "repo_id": "pirocheto/schp-pascal-7",
        "model_dir": ROOT / "schp-pascal-7",
        "onnx_files": [
            "schp-pascal-7.onnx",
            "schp-pascal-7.onnx.data",
            "schp-pascal-7-int8-static.onnx",
        ],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["atr", "lip", "pascal"], default="atr")
parser.add_argument("--private", action="store_true")
parser.add_argument("--no-onnx", action="store_true")
args = parser.parse_args()

cfg = _CONFIGS[args.dataset]
REPO_ID = cfg["repo_id"]
PRIVATE = args.private
INCLUDE_ONNX = not args.no_onnx
MODEL_DIR = str(cfg["model_dir"])
ONNX_DIR = str(ROOT / "onnx")
# ─────────────────────────────────────────────────────────────────────────────

api = HfApi()

print(f"Repo : {REPO_ID}  (private={PRIVATE})")
api.create_repo(repo_id=REPO_ID, private=PRIVATE, exist_ok=True, repo_type="model")

# ── Push model + processor ────────────────────────────────────────────────────
print("\n[1/3] Loading model...")
model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_DIR, trust_remote_code=True
)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("[2/3] Pushing model + processor...")
model.push_to_hub(REPO_ID, private=PRIVATE)
processor.push_to_hub(REPO_ID, private=PRIVATE)
print("      ✓ model.safetensors + config.json + preprocessor_config.json")

# Push source files (required for trust_remote_code)
for src_file in Path(MODEL_DIR).glob("*.py"):
    api.upload_file(
        path_or_fileobj=str(src_file),
        path_in_repo=src_file.name,
        repo_id=REPO_ID,
    )
    print(f"      ✓ {src_file.name}")

# ── Push ONNX files ──────────────────────────────────────────────────────────
if INCLUDE_ONNX:
    print("\n[3/3] Pushing ONNX files...")
    for fname in cfg["onnx_files"]:
        fpath = Path(ONNX_DIR) / fname
        if not fpath.exists():
            print(f"      ⚠ {fname} not found, skipping")
            continue
        size_mb = fpath.stat().st_size / 1024**2
        print(f"      uploading {fname} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f"onnx/{fname}",
            repo_id=REPO_ID,
        )
        print(f"      ✓ onnx/{fname}")
else:
    print("\n[3/3] ONNX skipped")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n✓ Model available at: https://huggingface.co/{REPO_ID}")
print("\nUsage from the Hub:")
print("  from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor")
print(
    f'  model     = AutoModelForSemanticSegmentation.from_pretrained("{REPO_ID}", trust_remote_code=True)'
)
print(
    f'  processor = AutoImageProcessor.from_pretrained("{REPO_ID}", trust_remote_code=True)'
)
