# SCHP — Self-Correction Human Parsing

Packaging of the [Self-Correction for Human Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) (SCHP) models for the 🤗 Transformers `AutoModel` API, with ONNX export and INT8 quantization.

Three checkpoints are available, each targeting a different dataset and granularity level:

| Model | Dataset | Classes | mIoU | HuggingFace Hub |
|-------|---------|---------|------|-----------------|
| `schp-atr-18` | ATR | 18 (clothing + body) | 82.29% | [pirocheto/schp-atr-18](https://huggingface.co/pirocheto/schp-atr-18) |
| `schp-lip-20` | LIP | 20 (clothing + body) | 59.36% | [pirocheto/schp-lip-20](https://huggingface.co/pirocheto/schp-lip-20) |
| `schp-pascal-7` | Pascal Person Part | 7 (coarse body parts) | 71.46% | [pirocheto/schp-pascal-7](https://huggingface.co/pirocheto/schp-pascal-7) |

---

## Overview

**SCHP** is a state-of-the-art human parsing architecture based on a **ResNet-101** backbone with a self-correction mechanism. It produces three outputs per forward pass:

| Output | Shape | Description |
|--------|-------|-------------|
| `logits` | `(B, C, H, W)` | Raw segmentation logits |
| `parsing_logits` | `(B, C, H, W)` | Refined parsing logits |
| `edge_logits` | `(B, 1 or 2, H, W)` | Edge prediction logits |

**Example use cases:**
- 🎨 **Outfit palette extraction** — mask each clothing region then cluster colors per garment
- 🏷️ **Product tagging for e-commerce** — automatically label photos with clothing categories
- 👚 **Virtual try-on pre-processing** — generate garment masks for VITON / LaDI-VTON
- ✏️ **Dataset annotation** — bootstrap labeling pipelines with predicted masks
- ✂️ **Clothing area cropping** — crop tight bounding boxes around specific items
- 🏃 **Body part segmentation** — segment coarse regions for pose-aware applications

---

## Models

### ATR — 18 classes

Trained on the **ATR** dataset (17 000+ images, fashion-focused).

| ID | Label | ID | Label | ID | Label |
|----|-------|----|-------|----|-------|
| 0 | Background | 6 | Pants | 12 | Left-leg |
| 1 | Hat | 7 | Dress | 13 | Right-leg |
| 2 | Hair | 8 | Belt | 14 | Left-arm |
| 3 | Sunglasses | 9 | Left-shoe | 15 | Right-arm |
| 4 | Upper-clothes | 10 | Right-shoe | 16 | Bag |
| 5 | Skirt | 11 | Face | 17 | Scarf |

### LIP — 20 classes

Trained on the **LIP** dataset (50 000+ images, real-world scenarios).

| ID | Label | ID | Label | ID | Label |
|----|-------|----|-------|----|-------|
| 0 | Background | 7 | Coat | 14 | Left-arm |
| 1 | Hat | 8 | Socks | 15 | Right-arm |
| 2 | Hair | 9 | Pants | 16 | Left-leg |
| 3 | Glove | 10 | Jumpsuits | 17 | Right-leg |
| 4 | Sunglasses | 11 | Scarf | 18 | Left-shoe |
| 5 | Upper-clothes | 12 | Skirt | 19 | Right-shoe |
| 6 | Dress | 13 | Face | | |

### Pascal Person Part — 7 classes

Trained on the **Pascal Person Part** dataset (3 000+ images, coarse body parts).

| ID | Label | ID | Label |
|----|-------|----|-------|
| 0 | Background | 4 | Lower Arms |
| 1 | Head | 5 | Upper Legs |
| 2 | Torso | 6 | Lower Legs |
| 3 | Upper Arms | | |

---

## Installation

```bash
git clone https://github.com/pirocheto/schp-human-parsing
cd schp-human-parsing
uv sync
```

---

## Quick Start

### PyTorch

```python
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch

# Choose one of: "pirocheto/schp-atr-18", "pirocheto/schp-lip-20", "pirocheto/schp-pascal-7"
model = AutoModelForSemanticSegmentation.from_pretrained(
    "pirocheto/schp-atr-18", trust_remote_code=True
)
processor = AutoImageProcessor.from_pretrained(
    "pirocheto/schp-atr-18", trust_remote_code=True
)

image = Image.open("photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# outputs.logits         — (1, 18, 512, 512)
# outputs.parsing_logits — (1, 18, 512, 512)
# outputs.edge_logits    — (1,  1, 512, 512)
seg_map = outputs.logits.argmax(dim=1).squeeze().numpy()  # (H, W)

# Map IDs back to label names
id2label = model.config.id2label
print(id2label[4])  # → "Upper-clothes"
```

You can also load from a local directory:

```python
model = AutoModelForSemanticSegmentation.from_pretrained("./schp-atr-18", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("./schp-atr-18", trust_remote_code=True)
```

### ONNX Runtime

Optimized ONNX files are available in the `onnx/` subfolder of each model directory.

| Model | File | Size | Notes |
|-------|------|------|-------|
| ATR | `schp-atr-18.onnx` | ~257 MB | FP32, dynamic batch |
| ATR | `schp-atr-18-int8-static.onnx` | ~66 MB | INT8, 99.94% pixel agreement |
| LIP | `schp-lip-20.onnx` | ~257 MB | FP32, dynamic batch |
| LIP | `schp-lip-20-int8-static.onnx` | ~66 MB | INT8, 99.09% pixel agreement |
| Pascal | `schp-pascal-7.onnx` | ~257 MB | FP32, dynamic batch |
| Pascal | `schp-pascal-7-int8-static.onnx` | ~66 MB | INT8, 99.77% pixel agreement |

```python
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor
from PIL import Image

model_path = hf_hub_download("pirocheto/schp-atr-18", "onnx/schp-atr-18-int8-static.onnx")
processor  = AutoImageProcessor.from_pretrained("pirocheto/schp-atr-18", trust_remote_code=True)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 8
sess = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])

image  = Image.open("photo.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="np")
logits = sess.run(["logits"], {"pixel_values": inputs["pixel_values"]})[0]
seg_map = logits.argmax(axis=1).squeeze()  # (H, W)
```

---

## Scripts

All scripts live in the `scripts/` directory.

### Download checkpoints

```bash
python scripts/download_checkpoints.py --dataset atr      # ATR checkpoint
python scripts/download_checkpoints.py --dataset lip      # LIP checkpoint
python scripts/download_checkpoints.py --dataset pascal   # Pascal checkpoint
python scripts/download_checkpoints.py --dataset all      # all checkpoints
```

### Download images

```bash
python scripts/download_images.py --n 10
```

### Convert an original `.pth` checkpoint

Convert an original SCHP `.pth` file to a Transformers-compatible model directory:

```bash
python scripts/convert_checkpoint.py --dataset atr --checkpoint checkpoints/exp-schp-201908261155-atr.pth
python scripts/convert_checkpoint.py --dataset lip --checkpoint checkpoints/exp-schp-201908261155-lip.pth
python scripts/convert_checkpoint.py --dataset pascal --checkpoint checkpoints/exp-schp-201908270938-pascal-person-part.pth
```

### Test a model

```bash
# PyTorch
python scripts/test_pytorch.py --model ./schp-atr-18 --image images/image_0.jpg

# ONNX
python scripts/test_onnx.py --model schp-atr-18/onnx/schp-atr-18-int8-static.onnx --image images/image_0.jpg  

# HuggingFace Hub
python scripts/test_hub.py --repo-id pirocheto/schp-lip-20
```

### Export to ONNX

```bash
python scripts/export_onnx.py --model ./schp-lip-20 --output schp-lip-20/onnx/schp-lip-20.onnx
```

### Quantize ONNX to INT8

```bash
# Dynamic quantization (no calibration data needed)
python scripts/quantize_onnx.py --mode dynamic

# Static quantization (better accuracy, requires calibration images)
python scripts/quantize_onnx.py --mode static --calib-images images/ --calib-n 10
```

### Benchmark PyTorch vs ONNX

```bash
python scripts/benchmark.py --dataset lip --image images/image_0.jpg --runs 20
```

### Test ONNX outputs

```bash
python scripts/test_onnx.py --model schp-atr-18/onnx/schp-atr-18-int8-static.onnx --image images/image_0.jpg
```

### Push to Hugging Face Hub

```bash
huggingface-cli login   # only once
python scripts/push_to_hub.py --dataset atr
python scripts/push_to_hub.py --dataset lip
python scripts/push_to_hub.py --dataset pascal
```

---

## Performance

Benchmarked on CPU (16-core, `intra_op_num_threads=8`):

### ATR (512×512 input)

| Backend | Latency | Speedup | Size |
|---------|---------|---------|------|
| PyTorch FP32 | ~430 ms | 1× | 256 MB |
| ONNX FP32 | ~293 ms | 1.5× | 257 MB |
| ONNX INT8 static | ~229 ms | **1.9×** | **66 MB** |

### LIP (473×473 input)

| Backend | Latency | Speedup | Size |
|---------|---------|---------|------|
| PyTorch FP32 | ~360 ms | 1× | 256 MB |
| ONNX FP32 | ~243 ms | 1.5× | 257 MB |
| ONNX INT8 static | ~189 ms | **1.9×** | **66 MB** |

### Pascal Person Part (512×512 input)

| Backend | Latency | Speedup | Size |
|---------|---------|---------|------|
| PyTorch FP32 | ~424 ms | 1× | 255 MB |
| ONNX FP32 | ~296 ms | 1.44× | 256 MB |
| ONNX INT8 static | ~218 ms | **1.94×** | **66 MB** |

---

## Citation

```bibtex
@article{li2020self,
  title={Self-Correction for Human Parsing},
  author={Li, Peike and Xu, Yunqiu and Wei, Yunchao and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  doi={10.1109/TPAMI.2020.3048039}
}
```

---

## License

MIT — see [LICENSE](Self-Correction-Human-Parsing/LICENSE).
