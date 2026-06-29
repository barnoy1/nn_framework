from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision

from infra.core import (
    build_label_id_remap_from_config_and_annotations,
    normalize_prediction_labels_for_metrics,
    to_canonical_predictions,
)
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.engine.flows.common.image_io import load_pil_image

from .eval_confusion import build_confusion_events
from .eval_vis_writer import write_sample_visualization


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
    gt_data: Optional[List[str]] = None,
    image_epoch_suffix: Optional[int],
    eval_vis_dir: Path,
    vis_logger,
    logger,
) -> Dict[str, object]:
    gt_modes = list(gt_data or [])
    transforms = build_image_preprocess_from_loader(
        app_config.engine.data.val_dataloader, logger=logger, default_size=640
    )
    label_id_remap = build_label_id_remap_from_config_and_annotations(
        remap_mscoco_category=bool(app_config.engine.data.remap_mscoco_category),
        class_id_to_name={
            int(k): str(v) for k, v in (app_config.engine.data.class_id_to_name or {}).items()
        },
        annotation_files=[
            str(dataset_pair.ann_file) for dataset_pair in app_config.engine.data.val_sets
        ],
    )
    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets_for_metric: List[Dict[str, torch.Tensor]] = []
    gt_total = 0
    gt_matched_iou50 = 0
    images_with_predictions = 0
    saved_vis = 0
    confusion_events: List[tuple[Optional[int], Optional[int]]] = []
    max_visualizations = max(0, int(vis_samples))
    first_conv_weight = next(
        (
            parameter
            for parameter in model_eval.parameters()
            if getattr(parameter, "ndim", 0) == 4
        ),
        None,
    )
    expected_channels = (
        int(first_conv_weight.shape[1]) if first_conv_weight is not None else None
    )
    visualization_indices: set[int] = set()
    if max_visualizations > 0 and len(samples) > 0:
        configured_seed = app_config.engine.train.seed
        try:
            rng_seed = int(configured_seed)
        except (TypeError, ValueError):
            rng_seed = 42
        rng = random.Random(rng_seed)
        selected_count = min(max_visualizations, len(samples))
        visualization_indices = set(rng.sample(range(len(samples)), selected_count))
    resolved_epoch = int(image_epoch_suffix) if image_epoch_suffix is not None else 0
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            original_image = load_pil_image(Path(sample["image_path"]))
            batch_tensor = transforms(original_image).unsqueeze(0).to(device)
            orig_sizes = torch.tensor(
                [[original_image.size[0], original_image.size[1]]], device=device
            )
            transformed_sizes = torch.tensor(
                [[int(batch_tensor.shape[-1]), int(batch_tensor.shape[-2])]],
                device=device,
            )
            if expected_channels and int(batch_tensor.shape[1]) != expected_channels:
                if expected_channels == 1:
                    batch_tensor = batch_tensor.mean(dim=1, keepdim=True)
                elif int(batch_tensor.shape[1]) == 1:
                    batch_tensor = batch_tensor.repeat(1, expected_channels, 1, 1)
                else:
                    batch_tensor = batch_tensor[:, :expected_channels, :, :]
            outputs = model_eval(batch_tensor)
            prediction = to_canonical_predictions(
                outputs, post_eval, orig_sizes, iou_types=app_config.engine.data.iou_types
            )[0]
            prediction_for_vis = to_canonical_predictions(
                outputs, post_eval, transformed_sizes, iou_types=app_config.engine.data.iou_types
            )[0]
            normalized_results = normalize_prediction_labels_for_metrics(
                [prediction, prediction_for_vis],
                label_id_remap=label_id_remap,
            )
            prediction = normalized_results[0]
            prediction_for_vis = normalized_results[1]
            raw_pred = {
                "labels": prediction["labels"].detach().cpu().long(),
                "boxes": prediction["boxes"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
            }

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
            if (
                sample["gt_labels"].numel() > 0
                and pred_for_metric["labels"].numel() > 0
            ):
                ious = torchvision.ops.box_iou(
                    pred_for_metric["boxes"], sample["gt_boxes"]
                )
                label_match = pred_for_metric["labels"].unsqueeze(1) == sample[
                    "gt_labels"
                ].unsqueeze(0)
                if label_match.any():
                    matched_ious = torch.where(
                        label_match, ious, torch.zeros_like(ious)
                    )
                    max_per_gt = matched_ious.max(dim=0).values
                    gt_matched_iou50 += int((max_per_gt >= 0.5).sum().item())

            confusion_events.extend(
                build_confusion_events(
                    pred_boxes=pred_for_metric["boxes"],
                    pred_labels=pred_for_metric["labels"],
                    gt_boxes=sample["gt_boxes"],
                    gt_labels=sample["gt_labels"],
                )
            )

            if sample_index in visualization_indices:
                write_sample_visualization(
                    sample=sample,
                    image_tensor=batch_tensor[0],
                    prediction_for_vis=prediction_for_vis,
                    class_id_to_name=class_id_to_name,
                    score_thr=score_thr,
                    eval_vis_dir=eval_vis_dir,
                    resolved_epoch=resolved_epoch,
                    vis_logger=vis_logger,
                    step=saved_vis,
                    gt_data=gt_modes,
                )
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
    vis_logger.log_text(
        tag="eval/detections_summary",
        text=f"visualized_samples={saved_vis} epoch={resolved_epoch}",
        step=resolved_epoch,
    )
    return {
        "all_predictions": all_predictions,
        "all_targets_for_metric": all_targets_for_metric,
        "confusion_events": confusion_events,
    }
