from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO

from infra.config import AppConfig
from infra.core import to_result_list
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.engine.evaluate import evaluate_predictions
from infra.utils.viz.visualize import render_prediction_with_yolo_caption
from infra.vis import create_visualization_logger


def build_eval_samples(val_sets, label_mapping: Dict[int, int]) -> List[Dict]:
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


def run_eval_artifacts(
    *,
    app_config: AppConfig,
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    logger,
    class_id_to_name: Dict[int, str],
    experiment_name: str,
    vis_samples: int = 16,
    score_thr: float = 0.3,
    image_epoch_suffix: Optional[int] = None,
    write_metrics_json: bool = True,
) -> Dict[str, float]:
    output_root = Path(app_config.train.output_dir)
    inference_dir = output_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    eval_vis_dir = inference_dir / "eval"
    eval_vis_dir.mkdir(parents=True, exist_ok=True)

    vis_logger = create_visualization_logger(
        output_root=output_root,
        experiment_name=experiment_name,
        tensorboard_enabled=bool(app_config.runtime.visualization.tensorboard.enabled),
        tensorboard_log_dir=str(app_config.runtime.visualization.tensorboard.log_dir),
        wandb_enabled=bool(app_config.runtime.visualization.wandb.enabled),
        wandb_dir=str(app_config.runtime.visualization.wandb.wandb_dir),
        wandb_entity=app_config.runtime.visualization.wandb.entity,
        logger_port=logger,
    )

    model_eval = model.deploy() if hasattr(model, "deploy") else model
    post_eval = postprocessor.deploy() if hasattr(postprocessor, "deploy") else postprocessor
    model_eval = model_eval.to(device).eval()

    transforms = build_image_preprocess_from_loader(app_config.data.val_dataloader, logger=logger, default_size=640)
    samples = build_eval_samples(app_config.data.val_sets, app_config.data.mapping or {})

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets_for_metric: List[Dict[str, torch.Tensor]] = []
    detection_records: List[Dict] = []

    gt_total = 0
    gt_matched_iou50 = 0
    images_with_predictions = 0
    saved_vis = 0

    suffix = ""
    if image_epoch_suffix is not None:
        suffix = f"_{int(image_epoch_suffix):04d}"

    with torch.no_grad():
        for sample in samples:
            original_image = Image.open(sample["image_path"]).convert("RGB")
            batch_tensor = transforms(original_image).unsqueeze(0).to(device)
            orig_sizes = torch.tensor([[original_image.size[0], original_image.size[1]]], device=device)

            outputs = model_eval(batch_tensor)
            prediction = to_result_list(outputs, post_eval, orig_sizes)[0]

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
            if score_thr > 0.0:
                keep = pred_for_metric["scores"] >= float(score_thr)
                pred_for_metric["boxes"] = pred_for_metric["boxes"][keep]
                pred_for_metric["scores"] = pred_for_metric["scores"][keep]
                pred_for_metric["labels"] = pred_for_metric["labels"][keep]
            if "masks" in prediction:
                pred_for_metric["masks"] = prediction["masks"].detach().cpu().bool()
                if score_thr > 0.0:
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

            if saved_vis < int(vis_samples):
                rendered = render_prediction_with_yolo_caption(
                    image=np.asarray(original_image.copy()),
                    prediction=prediction,
                    class_id_to_name=class_id_to_name,
                    confidence_threshold=score_thr,
                )
                image_name = f"eval_{saved_vis:05d}{suffix}.jpg"
                Image.fromarray(rendered).save(eval_vis_dir / image_name)
                vis_logger.log_image(tag="eval/visualization", image=rendered, step=saved_vis)
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

    detections_path = eval_vis_dir / "detections.json"
    with detections_path.open("w", encoding="utf-8") as file:
        json.dump(detection_records, file, indent=2)
    logger.info("Saved eval detections JSON to {}", detections_path)

    metrics = evaluate_predictions(
        predictions=all_predictions,
        targets=all_targets_for_metric,
        iou_types=app_config.data.iou_types,
    )

    if write_metrics_json:
        metrics_path = eval_vis_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump({key: float(value) for key, value in metrics.items()}, file, indent=2)
        logger.info("Saved eval metrics JSON to {}", metrics_path)

    vis_logger.log_metrics(metrics={f"eval/{key}": float(value) for key, value in metrics.items()}, step=0)
    vis_logger.close()

    return metrics
