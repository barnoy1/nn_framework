from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nn_framework.flows.common.runtime import build_flow_runtime
from nn_framework.utils.log import logger

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    return [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMG_EXTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework inference manager")
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--onnx-model", default="")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--allow-class-mismatch", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def _draw_boxes(image: Image.Image, labels, boxes, scores, score_thr: float) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(boxes):
        score = float(scores[index])
        if score < score_thr:
            continue
        draw.rectangle(list(box), outline="red", width=2)
        draw.text((box[0], box[1]), text=f"{int(labels[index])} {score:.3f}", fill="blue")
    return image


def _to_result_list(outputs, postprocessor, orig_sizes) -> List[dict]:
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs

    if isinstance(outputs, dict):
        processed = postprocessor(outputs, orig_sizes)
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        labels, boxes, scores = outputs
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": lab, "boxes": box, "scores": sco}
            for lab, box, sco in zip(labels, boxes, scores)
        ]
    else:
        processed = postprocessor(outputs, orig_sizes)

    if isinstance(processed, list):
        return processed

    if isinstance(processed, (tuple, list)) and len(processed) == 3:
        labels, boxes, scores = processed
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": lab, "boxes": box, "scores": sco}
            for lab, box, sco in zip(labels, boxes, scores)
        ]

    if isinstance(processed, dict) and {"labels", "boxes", "scores"}.issubset(set(processed.keys())):
        labels = processed["labels"]
        boxes = processed["boxes"]
        scores = processed["scores"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": lab, "boxes": box, "scores": sco}
            for lab, box, sco in zip(labels, boxes, scores)
        ]

    raise TypeError(f"Unsupported inference output format: {type(processed)}")


def _run_pytorch(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for PyTorch inference")

    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides, build_loaders=False)
    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(
        runtime.built.model,
        state,
        allow_mismatch=args.allow_class_mismatch,
    )
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    model = runtime.built.model
    postprocessor = runtime.built.postprocessor

    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    device = torch.device(args.device)
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = T.Compose([T.Resize((runtime.app_config.aug.image_size, runtime.app_config.aug.image_size)), T.ToTensor()])

    image_paths = list_images(args.input_dir)
    logger.info(
        "[mangr_inference] backend=pytorch device={} images={} input={}",
        args.device,
        len(image_paths),
        args.input_dir,
    )
    if not image_paths:
        logger.warning("[mangr_inference] no supported images found; nothing to process")
        return

    records = []
    processed = 0

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        original_images = [Image.open(path).convert("RGB") for path in batch_paths]
        batch_tensor = torch.stack([transforms(image) for image in original_images], dim=0).to(device)
        orig_sizes = torch.tensor([[image.size[0], image.size[1]] for image in original_images], device=device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            results = _to_result_list(outputs, postprocessor, orig_sizes)

        for image_path, image, result in zip(batch_paths, original_images, results):
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()

            rendered = _draw_boxes(image.copy(), labels, boxes, scores, score_thr=args.score_thr)
            rendered.save(output_dir / image_path.name)

            records.append(
                {
                    "image": image_path.name,
                    "labels": labels.tolist(),
                    "boxes": boxes.tolist(),
                    "scores": scores.tolist(),
                }
            )
            processed += 1

        logger.info("[mangr_inference] processed {}/{}", processed, len(image_paths))

    with (output_dir / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    logger.info("[mangr_inference] done. wrote {} images + {}", processed, output_dir / "detections.json")



def _run_onnx(args: argparse.Namespace) -> None:
    if not args.onnx_model:
        raise ValueError("--onnx-model is required for ONNX inference")

    import onnxruntime as ort

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(
        args.onnx_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    image_paths = list_images(args.input_dir)
    logger.info(
        "[mangr_inference] backend=onnx device={} images={} input={}",
        args.device,
        len(image_paths),
        args.input_dir,
    )
    if not image_paths:
        logger.warning("[mangr_inference] no supported images found; nothing to process")
        return

    records = []
    processed = 0

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        original_images = [Image.open(path).convert("RGB") for path in batch_paths]
        batch_tensor = torch.stack([transforms(image) for image in original_images], dim=0)
        orig_sizes = torch.tensor([[image.size[0], image.size[1]] for image in original_images], dtype=torch.int64)

        ort_outputs = session.run(
            None,
            {input_name: batch_tensor.numpy(), "orig_target_sizes": orig_sizes.numpy()},
        )

        labels_batch = ort_outputs[0]
        boxes_batch = ort_outputs[1]
        scores_batch = ort_outputs[2]

        for image_path, image, labels, boxes, scores in zip(batch_paths, original_images, labels_batch, boxes_batch, scores_batch):
            rendered = _draw_boxes(image.copy(), labels, boxes, scores, score_thr=args.score_thr)
            rendered.save(output_dir / image_path.name)

            records.append(
                {
                    "image": image_path.name,
                    "labels": np.asarray(labels).tolist(),
                    "boxes": np.asarray(boxes).tolist(),
                    "scores": np.asarray(scores).tolist(),
                }
            )
            processed += 1

        logger.info("[mangr_inference] processed {}/{}", processed, len(image_paths))

    with (output_dir / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    logger.info("[mangr_inference] done. wrote {} images: {}", processed, output_dir)


def main() -> None:
    args = parse_args()
    if args.onnx_model:
        _run_onnx(args)
    else:
        _run_pytorch(args)


if __name__ == "__main__":
    main()
