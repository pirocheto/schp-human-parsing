"""
benchmark.py — Compare inference speed: Transformers vs ONNX Runtime.

Usage:
    python scripts/benchmark.py --dataset atr
    python scripts/benchmark.py --dataset lip --image images/image_0.jpg --runs 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

ROOT = Path(__file__).parent.parent

_CONFIGS = {
    "atr": {
        "model_dir": ROOT / "schp-atr-18",
        "onnx_path": ROOT / "schp-atr-18" / "onnx" / "schp-atr-18.onnx",
    },
    "lip": {
        "model_dir": ROOT / "schp-lip-20",
        "onnx_path": ROOT / "schp-lip-20" / "onnx" / "schp-lip-20.onnx",
    },
}

IMAGE_PATH = str(ROOT / "images" / "image_0.jpg")
N_RUNS = 20
WARMUP = 3
ORT_THREADS = 8


def _make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def benchmark_transformers(
    model_dir: str, pixel_values: torch.Tensor, runs: int
) -> tuple[float, float]:
    print("\n── Transformers (PyTorch CPU) ──────────────────────────────────────")
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_dir, trust_remote_code=True
    )
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP):
            model(pixel_values=pixel_values)

    times = []
    with torch.no_grad():
        for i in range(runs):
            t0 = time.perf_counter()
            model(pixel_values=pixel_values)
            times.append(time.perf_counter() - t0)
            print(f"  run {i + 1:2d}/{runs}: {times[-1] * 1000:.1f} ms", end="\r")

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  mean: {mean_ms:.1f} ms  ±  {std_ms:.1f} ms  (over {runs} runs)    ")
    return mean_ms, std_ms


def benchmark_onnx(
    onnx_path: str, pixel_np: np.ndarray, runs: int
) -> tuple[float, float]:
    print("\n── ONNX Runtime (CPU) ─────────────────────────────────────────────")
    sess = _make_session(onnx_path)

    for _ in range(WARMUP):
        sess.run(None, {"pixel_values": pixel_np})

    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"pixel_values": pixel_np})
        times.append(time.perf_counter() - t0)
        print(f"  run {i + 1:2d}/{runs}: {times[-1] * 1000:.1f} ms", end="\r")

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  mean: {mean_ms:.1f} ms  ±  {std_ms:.1f} ms  (over {runs} runs)    ")
    return mean_ms, std_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["atr", "lip"], default="atr")
    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    args = parser.parse_args()

    cfg = _CONFIGS[args.dataset]
    model_dir = str(cfg["model_dir"])
    onnx_path = str(cfg["onnx_path"])

    print(f"Dataset: {args.dataset}")
    print(f"Image  : {args.image}")
    print(f"Runs   : {args.runs}  (+ {WARMUP} warmup)")

    # ── Preprocess once (shared) ─────────────────────────────────────────────
    image = Image.open(args.image).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values_pt: torch.Tensor = inputs["pixel_values"]
    pixel_values_np: np.ndarray = pixel_values_pt.numpy()

    # ── Run benchmarks ───────────────────────────────────────────────────────
    pt_mean, pt_std = benchmark_transformers(model_dir, pixel_values_pt, args.runs)
    ort_mean, ort_std = benchmark_onnx(onnx_path, pixel_values_np, args.runs)

    # ── File sizes ───────────────────────────────────────────────────────────
    def _size_mb(path: str) -> str:
        p = Path(path)
        total = p.stat().st_size if p.exists() else 0
        data = p.with_suffix(".onnx.data")
        if data.exists():
            total += data.stat().st_size
        return f"{total / 1024**2:.0f} MB" if total else "n/a"

    safetensors = Path(model_dir) / "model.safetensors"
    pt_size = _size_mb(str(safetensors))
    ort_size = _size_mb(onnx_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    speedup = pt_mean / ort_mean
    print("\n" + "═" * 60)
    print(f"  {'Backend':<20} {'Latency':>12}  {'Size':>8}")
    print(f"  {'-'*20} {'-'*12}  {'-'*8}")
    print(f"  {'Transformers (FP32)':<20} {pt_mean:7.1f} ms      {pt_size:>8}")
    print(f"  {'ONNX Runtime (FP32)':<20} {ort_mean:7.1f} ms      {ort_size:>8}")
    print(f"  {'Speedup':<20} {speedup:>7.2f}×")
    print("═" * 60)


if __name__ == "__main__":
    main()
