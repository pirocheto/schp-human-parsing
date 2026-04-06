"""
convert_checkpoint.py — Convert an original SCHP .pth file to a
Transformers-compatible model directory (PyTorch bin + config).

Usage:
    python scripts/convert_checkpoint.py --dataset atr --checkpoint exp-schp-201908301523-atr.pth
    python scripts/convert_checkpoint.py --dataset lip --checkpoint exp-schp-201908261155-lip.pth

    # Override output directory:
    python scripts/convert_checkpoint.py --dataset lip --checkpoint ... --output my-dir

The output directory can then be used with:
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    model = AutoModelForSemanticSegmentation.from_pretrained("./schp-lip-20", trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained("./schp-lip-20", trust_remote_code=True)
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent.parent

_DATASETS = {
    "atr": {"output": "schp-atr-18"},
    "lip": {"output": "schp-lip-20"},
}

_AUTO_MAP = {
    "AutoConfig": "configuration_schp.SCHPConfig",
    "AutoModelForSemanticSegmentation": "modeling_schp.SCHPForSemanticSegmentation",
}

_PROCESSOR_AUTO_MAP = {
    "AutoImageProcessor": "image_processing_schp.SCHPImageProcessor",
}

_SOURCE_FILES = [
    "configuration_schp.py",
    "modeling_schp.py",
    "image_processing_schp.py",
]


def _load_modules(model_dir: Path):
    """Load the three SCHP Python files from model_dir into sys.modules."""
    # Create a 'schp' package so internal 'from schp.xxx import' works
    schp_pkg = types.ModuleType("schp")
    sys.modules["schp"] = schp_pkg

    for name in ("configuration_schp", "image_processing_schp", "modeling_schp"):
        spec = importlib.util.spec_from_file_location(name, model_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        sys.modules[f"schp.{name}"] = mod
        setattr(schp_pkg, name, mod)
        spec.loader.exec_module(mod)

    return (
        sys.modules["configuration_schp"].SCHPConfig,
        sys.modules["image_processing_schp"].SCHPImageProcessor,
        sys.modules["modeling_schp"].SCHPForSemanticSegmentation,
    )


def convert(checkpoint_path: str, output_dir: str, dataset: str) -> None:
    model_dir = ROOT / _DATASETS[dataset]["output"]
    SCHPConfig, SCHPImageProcessor, SCHPForSemanticSegmentation = _load_modules(
        model_dir
    )

    print(f"Dataset  : {dataset}")
    print(f"Source   : {model_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    config = SCHPConfig()
    model = SCHPForSemanticSegmentation.from_schp_checkpoint(
        checkpoint_path,
        config=config,
        map_location="cpu",
    )
    model.eval()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {out}")
    model.save_pretrained(str(out))

    processor = SCHPImageProcessor()
    processor.save_pretrained(str(out))

    # ── Patch auto_map into config files ─────────────────────────────────────
    config_path = out / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["auto_map"] = _AUTO_MAP
    config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

    proc_path = out / "preprocessor_config.json"
    if proc_path.exists():
        pc = json.loads(proc_path.read_text())
        pc["auto_map"] = _PROCESSOR_AUTO_MAP
        proc_path.write_text(json.dumps(pc, indent=2, sort_keys=True))

    print("\nDone. To use:")
    print(
        "  from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor"
    )
    print(
        f'  model = AutoModelForSemanticSegmentation.from_pretrained("{out}", trust_remote_code=True)'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SCHP .pth checkpoint to Transformers format."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(_DATASETS),
        help="Dataset variant: 'atr' (18 classes) or 'lip' (20 classes).",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: schp-atr-18 or schp-lip-20).",
    )
    args = parser.parse_args()

    output = args.output or str(ROOT / _DATASETS[args.dataset]["output"])
    convert(args.checkpoint, output, args.dataset)
