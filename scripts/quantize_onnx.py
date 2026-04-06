"""
quantize_onnx.py — Quantize the SCHP ONNX model.

Two modes:
  --mode dynamic  (default) — INT8 weights only, no calibration data needed.
                               Fastest to run, ~4× smaller weights.
  --mode static             — INT8 weights + activations, needs calibration images.
                               Slower to prepare but faster inference.

Usage:
    python quantize_onnx.py
    python quantize_onnx.py --mode dynamic
    python quantize_onnx.py --mode static --calib-images images/ --calib-n 10
    python quantize_onnx.py --input onnx/schp-atr-18.onnx --output onnx/schp-atr-18-int8.onnx
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent.parent

_CONFIGS = {
    "atr": {
        "model_dir": ROOT / "schp-atr-18",
        "onnx_input": ROOT / "schp-atr-18" / "onnx" / "schp-atr-18.onnx",
        "onnx_dynamic": ROOT / "schp-atr-18" / "onnx" / "schp-atr-18-int8-dynamic.onnx",
        "onnx_static": ROOT / "schp-atr-18" / "onnx" / "schp-atr-18-int8-static.onnx",
    },
    "lip": {
        "model_dir": ROOT / "schp-lip-20",
        "onnx_input": ROOT / "schp-lip-20" / "onnx" / "schp-lip-20.onnx",
        "onnx_dynamic": ROOT / "schp-lip-20" / "onnx" / "schp-lip-20-int8-dynamic.onnx",
        "onnx_static": ROOT / "schp-lip-20" / "onnx" / "schp-lip-20-int8-static.onnx",
    },
}


# ── Calibration dataset (for static quantization) ────────────────────────────


class CalibrationDataReader:
    """Feeds preprocessed images to the static calibration pipeline."""

    def __init__(self, image_dir: str, n: int, model_dir: str) -> None:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        )
        paths = (
            sorted(Path(image_dir).glob("*.jpg"))[:n]
            + sorted(Path(image_dir).glob("*.png"))[:n]
        )
        paths = paths[:n]
        if not paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        print(f"  Calibration: {len(paths)} images from {image_dir}")

        self._data = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            out = processor(images=img, return_tensors="np")
            self._data.append({"pixel_values": out["pixel_values"]})
        self._idx = 0

    def get_next(self) -> dict | None:
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item


# ── Dynamic quantization ──────────────────────────────────────────────────────


def quantize_dynamic(input_path: str, output_path: str) -> None:
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_dynamic as _quantize_dynamic

    print("\n── Dynamic INT8 quantization ───────────────────────────────────────")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")

    t0 = time.perf_counter()
    _quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        use_external_data_format=False,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")


# ── Static quantization ───────────────────────────────────────────────────────


def quantize_static(
    input_path: str, output_path: str, image_dir: str, n: int, model_dir: str = ""
) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static as _quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Step 1: pre-process (shape inference + model optimisation)
    preprocessed = input_path.replace(".onnx", "-preprocessed.onnx")
    print("\n── Static INT8 quantization ────────────────────────────────────────")
    print("  Pre-processing model...")
    quant_pre_process(
        input_path,
        preprocessed,
        skip_optimization=False,
        skip_symbolic_shape=True,  # dynamo-exported models trip on symbolic shape inference
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    )

    # Step 2: calibrate + quantize
    print("  Calibrating...")
    reader = CalibrationDataReader(image_dir, n, model_dir)

    t0 = time.perf_counter()
    _quantize_static(
        model_input=preprocessed,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    Path(preprocessed).unlink(missing_ok=True)
    data_file = Path(preprocessed.replace(".onnx", ".onnx.data"))
    data_file.unlink(missing_ok=True)


# ── Verify + benchmark ────────────────────────────────────────────────────────


def verify(
    original_path: str,
    quantized_path: str,
    model_dir: str,
    image_path: str = str(ROOT / "images" / "image_0.jpg"),
) -> None:
    from transformers import AutoImageProcessor

    print("\n── Verification ────────────────────────────────────────────────────")
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="np")["pixel_values"]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_orig = ort.InferenceSession(
        original_path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    sess_quant = ort.InferenceSession(
        quantized_path, sess_options=so, providers=["CPUExecutionProvider"]
    )

    out_orig = sess_orig.run(None, {"pixel_values": pixel_values})[0]
    out_quant = sess_quant.run(None, {"pixel_values": pixel_values})[0]

    # Accuracy: % of pixels with same argmax prediction
    pred_orig = np.argmax(out_orig[0], axis=0)
    pred_quant = np.argmax(out_quant[0], axis=0)
    agreement = np.mean(pred_orig == pred_quant) * 100
    print(f"  Pixel agreement (argmax): {agreement:.2f}%")

    # Speed comparison (10 runs each)
    N = 10
    times_orig, times_quant = [], []
    for _ in range(3):  # warmup
        sess_orig.run(None, {"pixel_values": pixel_values})
        sess_quant.run(None, {"pixel_values": pixel_values})
    for _ in range(N):
        t0 = time.perf_counter()
        sess_orig.run(None, {"pixel_values": pixel_values})
        times_orig.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sess_quant.run(None, {"pixel_values": pixel_values})
        times_quant.append(time.perf_counter() - t0)

    orig_ms = np.mean(times_orig) * 1000
    quant_ms = np.mean(times_quant) * 1000
    speedup = orig_ms / quant_ms

    # File sizes
    def _size(path: str) -> float:
        p = Path(path)
        total = p.stat().st_size
        data = p.with_suffix(".onnx.data")
        if data.exists():
            total += data.stat().st_size
        return total / 1024**2

    print(f"\n  {'':20s}  {'Latency':>10s}  {'Size':>8s}")
    print(
        f"  {'FP32 (original)':20s}  {orig_ms:>9.1f}ms  {_size(original_path):>6.1f} MB"
    )
    print(
        f"  {'INT8 (quantized)':20s}  {quant_ms:>9.1f}ms  {_size(quantized_path):>6.1f} MB"
    )
    print(f"  {'Speedup':20s}  {speedup:>9.2f}×")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["atr", "lip"], default="atr")
    parser.add_argument("--mode", choices=["dynamic", "static"], default="static")
    parser.add_argument("--input", default=None, help="Override input ONNX path")
    parser.add_argument("--output", default=None, help="Override output ONNX path")
    parser.add_argument(
        "--calib-images",
        default=str(ROOT / "images"),
        help="Folder with calibration images (static only)",
    )
    parser.add_argument(
        "--calib-n",
        type=int,
        default=10,
        help="Number of calibration images (static only)",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    cfg = _CONFIGS[args.dataset]
    model_dir = str(cfg["model_dir"])
    input_path = args.input or str(cfg["onnx_input"])
    output = args.output or (
        str(cfg["onnx_dynamic"]) if args.mode == "dynamic" else str(cfg["onnx_static"])
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dynamic":
        quantize_dynamic(input_path, output)
    else:
        quantize_static(input_path, output, args.calib_images, args.calib_n, model_dir)

    if not args.no_verify:
        verify(input_path, output, model_dir)


if __name__ == "__main__":
    main()
