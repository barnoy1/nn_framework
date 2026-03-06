from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
from PIL import Image

from infra.core import to_result_list
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.engine.flows.common.image_io import load_pil_image
from infra.utils.viz.visualize import render_prediction_with_yolo_caption


def run_eval_inference_loop(
    *,
    app_config,
    samples,
    model_eval,
    post_eval,
    device: torch.device,
    class_id_to_name: Dict[int, str],
    score_thr: float,
    vis_samples: int,
    image_epoch_suffix: Optional[int],
    eval_vis_dir: Path,
    vis_logger,
    logger,
) -> Dict[str, object]:
    transforms = build_image_preprocess_from_loader(app_config.data.val_dataloader, logger=logger, default_size=640)

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets_for_metric: List[Dict[str, torch.Tensor]] = []
    detection_records: List[Dict] = []

    gt_total = 0
    gt_matched_iou50 = 0
    images_with_predictions = 0
    saved_vis = 0
    confusion_events: List[tuple[Optional[int], Optional[int]]] = []

    suffix = ""
    if image_epoch_suffix is not None:
        suffix = f"_{int(image_epoch_suffix):04d}"

    with torch.no_grad():
        for sample in samples:
            original_image = load_pil_image(Path(sample["image_path"]))
            batch_tensor = transforms(original_image).unsqueeze(0).to(device)
            orig_sizes = torch.tensor([[original_image.size[0], original_image.size[1]]], device=device)
            transformed_sizes = torch.tensor(
                [[int(batch_tensor.shape[-1]), int(batch_tensor.shape[-2])]],
                device=device,
            )

            outputs = model_eval(batch_tensor)
            prediction = to_result_list(outputs, post_eval, orig_sizes)[0]
            prediction_for_vis = to_result_list(outputs, post_eval, transformed_sizes)[0]

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

            pred_labels = pred_for_metric["labels"].detach().cpu().long()
            gt_labels = sample["gt_labels"].detach().cpu().long()
            if pred_for_metric["boxes"].numel() > 0 and sample["gt_boxes"].numel() > 0:
                ious_for_match = torchvision.ops.box_iou(pred_for_metric["boxes"], sample["gt_boxes"])
                candidate_pairs = []
                for pred_idx in range(ious_for_match.shape[0]):
                    for gt_idx in range(ious_for_match.shape[1]):
                        iou_value = float(ious_for_match[pred_idx, gt_idx].item())
                        if iou_value >= 0.5:
                            candidate_pairs.append((iou_value, pred_idx, gt_idx))
                candidate_pairs.sort(key=lambda item: item[0], reverse=True)
                matched_pred = set()
                matched_gt = set()
                for _, pred_idx, gt_idx in candidate_pairs:
                    if pred_idx in matched_pred or gt_idx in matched_gt:
                        continue
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)
                    confusion_events.append((int(gt_labels[gt_idx].item()), int(pred_labels[pred_idx].item())))
                for gt_idx in range(gt_labels.numel()):
                    if gt_idx not in matched_gt:
                        confusion_events.append((int(gt_labels[gt_idx].item()), None))
                for pred_idx in range(pred_labels.numel()):
                    if pred_idx not in matched_pred:
                        confusion_events.append((None, int(pred_labels[pred_idx].item())))
            else:
                for gt_idx in range(gt_labels.numel()):
                    confusion_events.append((int(gt_labels[gt_idx].item()), None))
                for pred_idx in range(pred_labels.numel()):
                    confusion_events.append((None, int(pred_labels[pred_idx].item())))

            if saved_vis < int(vis_samples):
                vis_tensor = batch_tensor[0].detach().cpu()
                if vis_tensor.ndim != 3:
                    raise ValueError(f"Expected CHW tensor for visualization, got shape={tuple(vis_tensor.shape)}")

                if int(vis_tensor.shape[0]) == 1:
                    vis_tensor = vis_tensor.repeat(3, 1, 1)
                elif int(vis_tensor.shape[0]) > 3:
                    vis_tensor = vis_tensor[:3]

                vis_image = (vis_tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                rendered = render_prediction_with_yolo_caption(
                    image=vis_image,
                    prediction=prediction_for_vis,
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
    vis_logger.log_text(
        tag="eval/detections_summary",
        text=f"records={len(detection_records)} path={detections_path}",
        step=0,
    )

    return {
        "all_predictions": all_predictions,
        "all_targets_for_metric": all_targets_for_metric,
        "confusion_events": confusion_events,
    }
