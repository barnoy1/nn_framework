from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.adapters import LoguruLoggerAdapter
from infra.core import to_result_list
from infra.engine.evaluate import evaluate_predictions
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.engine.flows.eval.dataset_profile import model_num_classes, profile_dataset_distribution
from infra.utils.viz.visualize import render_prediction_with_yolo_caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework evaluation manager")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vis-samples", type=int, default=16)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def _build_eval_samples(val_sets, label_mapping: Dict[int, int]) -> List[Dict]:
    samples: List[Dict] = []
    normalized_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}

    for dataset_pair in val_sets:
        coco = COCO(str(dataset_pair.ann_file))
        image_ids = sorted(coco.getImgIds())
        categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda cat: cat["id"])
        category_id_to_contiguous = {cat["id"]: idx for idx, cat in enumerate(categories)}

        for image_id in image_ids:
            image_meta = coco.loadImgs([image_id])[0]
            image_path = Path(dataset_pair.img_dir) / image_meta["file_name"]
            width = int(image_meta["width"])
            height = int(image_meta["height"])

            ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            annotations = coco.loadAnns(ann_ids)

            boxes_xyxy: List[List[float]] = []
            labels: List[int] = []
            for ann in annotations:
                if ann.get("iscrowd", 0) == 1:
                    continue
                category_id = ann.get("category_id")
                if category_id not in category_id_to_contiguous:
                    continue

                x, y, w, h = ann["bbox"]
                x1 = float(max(0.0, x))
                y1 = float(max(0.0, y))
                x2 = float(min(width, x + w))
                y2 = float(min(height, y + h))
                if x2 <= x1 or y2 <= y1:
                    continue

                contiguous_label = int(category_id_to_contiguous[category_id])
                mapped_label = int(normalized_mapping.get(contiguous_label, contiguous_label))

                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(mapped_label)

            if boxes_xyxy:
                gt_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
                gt_labels = torch.tensor(labels, dtype=torch.long)
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.long)

            samples.append(
                {
                    "image_id": int(image_id),
                    "image_path": image_path,
                    "file_name": str(image_meta["file_name"]),
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels,
                }
            )

    return samples


def _build_eval_preprocess(val_loader_cfg: Dict | None, logger) -> T.Compose:
    ops = []
    dataset_cfg = val_loader_cfg.get("dataset") if isinstance(val_loader_cfg, dict) else None
    transforms_cfg = dataset_cfg.get("transforms") if isinstance(dataset_cfg, dict) else None
    configured_ops = transforms_cfg.get("ops") if isinstance(transforms_cfg, dict) else None

    if isinstance(configured_ops, list):
        for op in configured_ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type", "")).strip().lower()

            if op_type == "resize":
                size = op.get("size")
                if isinstance(size, list) and len(size) >= 2:
                    height = int(size[0])
                    width = int(size[1])
                else:
                    height = int(size)
                    width = int(size)
                ops.append(T.Resize((height, width)))
                continue

            if op_type == "convertpilimage":
                continue

            logger.warning("Unsupported val_dataloader transform '{}' in eval preprocessing; skipping.", op.get("type"))

    ops.append(T.ToTensor())
    return T.Compose(ops)


def main() -> None:
    args = parse_args()
    logger = LoguruLoggerAdapter()
    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides, config_path=args.config)
    runtime.app_config.train.use_ema = False
    runtime.app_config.train.mixed_precision = "no"
    runtime.built.ema_model = None

    profile_dataset_distribution(runtime, logger)

    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    net_classes = model_num_classes(runtime.built.model)
    if net_classes is not None:
        configured_mapping = runtime.app_config.data.mapping or {}
        configured_label_ids = sorted({int(label_id) for label_id in configured_mapping.values()})
        if configured_label_ids:
            out_of_range = [label_id for label_id in configured_label_ids if label_id < 0 or label_id >= net_classes]
            if out_of_range:
                logger.warning(
                    "validation label ids {} are out of model class range [0, {}]. Evaluation AP will be invalid.",
                    out_of_range,
                    net_classes - 1,
                )
            else:
                logger.info(
                    "Validation uses mapped label ids {} within model class range [0, {}].",
                    configured_label_ids,
                    net_classes - 1,
                )

    output_root = Path(runtime.app_config.train.output_dir)
    inference_dir = output_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    eval_vis_dir = inference_dir / "eval"
    eval_vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = runtime.built.model
    postprocessor = runtime.built.postprocessor
    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()
    model = model.to(device).eval()
    class_id_to_name = runtime.built.class_id_to_name

    transforms = _build_eval_preprocess(runtime.app_config.data.val_dataloader, logger)
    batch_size = 1

    samples = _build_eval_samples(runtime.app_config.data.val_sets, runtime.app_config.data.mapping or {})

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets_for_metric: List[Dict[str, torch.Tensor]] = []
    detection_records: List[Dict] = []

    gt_total = 0
    gt_matched_iou50 = 0
    images_with_predictions = 0
    saved_vis = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            original_images = [Image.open(sample["image_path"]).convert("RGB") for sample in batch_samples]

            batch_tensor = torch.stack([transforms(image) for image in original_images], dim=0).to(device)
            orig_sizes = torch.tensor([[image.size[0], image.size[1]] for image in original_images], device=device)

            outputs = model(batch_tensor)
            results = to_result_list(outputs, postprocessor, orig_sizes)

            for sample, original_image, prediction in zip(batch_samples, original_images, results):
                raw_pred = {
                    "labels": prediction["labels"].detach().cpu().long(),
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                }

                detection_records.append(
                    {
                        "image": sample["file_name"],
                        "labels": raw_pred["labels"].tolist(),
                        "boxes": raw_pred["boxes"].tolist(),
                        "scores": raw_pred["scores"].tolist(),
                    }
                )

                pred_for_metric = {
                    "boxes": raw_pred["boxes"],
                    "scores": raw_pred["scores"],
                    "labels": raw_pred["labels"],
                }
                if args.score_thr > 0.0:
                    keep = pred_for_metric["scores"] >= float(args.score_thr)
                    pred_for_metric["boxes"] = pred_for_metric["boxes"][keep]
                    pred_for_metric["scores"] = pred_for_metric["scores"][keep]
                    pred_for_metric["labels"] = pred_for_metric["labels"][keep]
                if "masks" in prediction:
                    pred_for_metric["masks"] = prediction["masks"].detach().cpu().bool()
                    if args.score_thr > 0.0:
                        pred_for_metric["masks"] = pred_for_metric["masks"][keep]

                all_predictions.append(pred_for_metric)
                target_for_metric = {
                    "boxes": sample["gt_boxes"],
                    "labels": sample["gt_labels"],
                }
                all_targets_for_metric.append(target_for_metric)

                if pred_for_metric["labels"].numel() > 0:
                    images_with_predictions += 1

                gt_total += int(sample["gt_labels"].numel())
                if sample["gt_labels"].numel() > 0 and pred_for_metric["labels"].numel() > 0:
                    ious = torchvision.ops.box_iou(pred_for_metric["boxes"], sample["gt_boxes"])
                    pred_labels = pred_for_metric["labels"].unsqueeze(1)
                    gt_labels = sample["gt_labels"].unsqueeze(0)
                    label_match = pred_labels == gt_labels
                    if label_match.any():
                        matched_ious = torch.where(label_match, ious, torch.zeros_like(ious))
                        max_per_gt = matched_ious.max(dim=0).values
                        gt_matched_iou50 += int((max_per_gt >= 0.5).sum().item())

                if saved_vis < args.vis_samples:
                    rendered = render_prediction_with_yolo_caption(
                        image=np.asarray(original_image.copy()),
                        prediction=prediction,
                        class_id_to_name=class_id_to_name,
                        confidence_threshold=args.score_thr,
                    )
                    Image.fromarray(rendered).save(eval_vis_dir / f"eval_{saved_vis:05d}.jpg")
                    saved_vis += 1

    logger.info("Saved {} eval visualizations to {}", saved_vis, eval_vis_dir)

    if gt_total > 0:
        logger.info(
            "Eval diagnostic: images_with_predictions={} gt_total={} gt_matched_iou50={} recall50={:.4f}",
            images_with_predictions,
            gt_total,
            gt_matched_iou50,
            gt_matched_iou50 / float(gt_total),
        )

    detections_path = inference_dir / "detections.json"
    with detections_path.open("w", encoding="utf-8") as file:
        json.dump(detection_records, file, indent=2)
    logger.info("Saved eval detections JSON to {}", detections_path)

    metrics = evaluate_predictions(
        predictions=all_predictions,
        targets=all_targets_for_metric,
        iou_types=runtime.app_config.data.iou_types,
    )

    metrics_path = output_root / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump({key: float(value) for key, value in metrics.items()}, file, indent=2)
    logger.info("Saved eval metrics JSON to {}", metrics_path)

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("{}: {}", key, value)


if __name__ == "__main__":
    main()
