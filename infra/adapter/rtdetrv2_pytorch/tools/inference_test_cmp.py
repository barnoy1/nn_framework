from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from PIL import Image


def _add_project_root_to_sys_path() -> None:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            project_root = str(parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            return


_add_project_root_to_sys_path()


from infra.common.rendering.visualize import render_prediction_with_yolo_caption
from infra.core import to_result_list
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.engine.flows.common.runtime import build_flow_runtime

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_MODEL_PATH = (SCRIPT_DIR / ".." / "models" / "rtdetrv2_r18vd_3ch.onnx").resolve()
DEFAULT_CONFIG = (
    SCRIPT_DIR
    / ".."
    / "experiments"
    / "rtdetrv2_r18vd_120e_coco_instance_seg_rle.yaml"
).resolve()
SOURCE_OUTPUT = SCRIPT_DIR / "annotated_image_source.jpg"
ONNX_OUTPUT = SCRIPT_DIR / "annotated_image_onnx.jpg"
REPORT_OUTPUT = SCRIPT_DIR / "comparison_report.json"
SCORE_THRESHOLD = 0.5


def _prediction_to_topk_payload(prediction: dict, class_id_to_name: dict[int, str], top_k: int = 10) -> list[dict]:
    labels = np.asarray(prediction["labels"])
    boxes = np.asarray(prediction["boxes"])
    scores = np.asarray(prediction["scores"])
    if scores.size == 0:
        return []

    order = np.argsort(-scores)[:top_k]
    payload = []
    for idx in order:
        label_id = int(labels[idx])
        payload.append(
            {
                "label": class_id_to_name.get(label_id, str(label_id)),
                "score": float(scores[idx]),
                "bbox_xyxy": [float(v) for v in boxes[idx].tolist()],
            }
        )
    return payload


def _save_rendered(image: Image.Image, prediction: dict, class_id_to_name: dict[int, str], output_path: Path) -> None:
    rendered = render_prediction_with_yolo_caption(
        image=np.asarray(image.convert("RGB")),
        prediction=prediction,
        class_id_to_name=class_id_to_name,
        confidence_threshold=SCORE_THRESHOLD,
    )
    Image.fromarray(rendered).save(output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx-model", type=str, default=str(ONNX_MODEL_PATH))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    runtime = build_flow_runtime(overrides=[], config_path=args.config, build_loaders=False)
    class_id_to_name = runtime.built.class_id_to_name

    state = runtime.wrapper.load_checkpoint_state(str(Path(args.checkpoint).resolve()))
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    runtime.wrapper.safe_load_state_dict(runtime.built.model, state)

    model = runtime.built.model
    postprocessor = runtime.built.postprocessor
    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    device = torch.device(args.device)
    model.to(device).eval()

    response = requests.get("https://media.roboflow.com/dog.jpg", timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    transforms = build_image_preprocess_from_loader(runtime.app_config.data.val_dataloader, default_size=640)
    image_tensor = transforms(image).unsqueeze(0).to(device)
    orig_sizes = torch.tensor([[image.size[0], image.size[1]]], dtype=torch.int64, device=device)

    with torch.no_grad():
        source_outputs = model(image_tensor)
        source_results = to_result_list(source_outputs, postprocessor, orig_sizes)
    source_prediction = {
        "labels": source_results[0]["labels"].detach().cpu().numpy(),
        "boxes": source_results[0]["boxes"].detach().cpu().numpy(),
        "scores": source_results[0]["scores"].detach().cpu().numpy(),
    }
    _save_rendered(image, source_prediction, class_id_to_name, SOURCE_OUTPUT)

    onnx_session = ort.InferenceSession(
        str(Path(args.onnx_model).resolve()),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"],
    )
    onnx_labels, onnx_boxes, onnx_scores = onnx_session.run(
        None,
        {
            "images": image_tensor.detach().cpu().numpy(),
            "orig_target_sizes": orig_sizes.detach().cpu().numpy(),
        },
    )
    onnx_prediction = {
        "labels": np.asarray(onnx_labels[0]),
        "boxes": np.asarray(onnx_boxes[0]),
        "scores": np.asarray(onnx_scores[0]),
    }
    _save_rendered(image, onnx_prediction, class_id_to_name, ONNX_OUTPUT)

    report = {
        "threshold": SCORE_THRESHOLD,
        "source_model": {
            "name": "RTDETRv2 (PyTorch)",
            "num_detections": int(np.sum(source_prediction["scores"] >= SCORE_THRESHOLD)),
            "top_detections": _prediction_to_topk_payload(source_prediction, class_id_to_name),
        },
        "onnx_model": {
            "path": str(Path(args.onnx_model).resolve()),
            "num_detections": int(np.sum(onnx_prediction["scores"] >= SCORE_THRESHOLD)),
            "top_detections": _prediction_to_topk_payload(onnx_prediction, class_id_to_name),
        },
        "outputs": {
            "source_image": str(SOURCE_OUTPUT),
            "onnx_image": str(ONNX_OUTPUT),
        },
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {SOURCE_OUTPUT}, {ONNX_OUTPUT}, {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
