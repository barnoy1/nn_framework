from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "raw_models" / "RT-DETR" / "rtdetrv2_pytorch"

ACTION_TO_RUNTIME_SECTION = {
    "train": "train",
    "eval": "eval",
    "inference": "inference",
    "inference-onnx": "inference_onnx",
    "export-onnx": "export_onnx",
    "export-coco-rle": "export_coco_rle",
}
