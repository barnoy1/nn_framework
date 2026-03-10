from __future__ import annotations

from typing import Dict, List

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@torch.no_grad()
def evaluate_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_types: List[str],
) -> Dict[str, float]:
    output: Dict[str, float] = {}

    bbox_metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=False
    )
    bbox_metric.update(predictions, targets)
    bbox_result = bbox_metric.compute()
    output.update(
        {
            "bbox_map": float(bbox_result.get("map", torch.tensor(0.0)).item()),
            "bbox_map_50": float(bbox_result.get("map_50", torch.tensor(0.0)).item()),
            "bbox_map_75": float(bbox_result.get("map_75", torch.tensor(0.0)).item()),
            "bbox_mar_100": float(bbox_result.get("mar_100", torch.tensor(0.0)).item()),
            "map": float(bbox_result.get("map", torch.tensor(0.0)).item()),
        }
    )

    if "segm" in iou_types:
        has_masks = all(
            ("masks" in prediction and "masks" in target)
            for prediction, target in zip(predictions, targets)
        )
        if has_masks:
            segm_metric = MeanAveragePrecision(
                box_format="xyxy", iou_type="segm", class_metrics=False
            )
            segm_metric.update(predictions, targets)
            segm_result = segm_metric.compute()
            output.update(
                {
                    "segm_map": float(segm_result.get("map", torch.tensor(0.0)).item()),
                    "segm_map_50": float(
                        segm_result.get("map_50", torch.tensor(0.0)).item()
                    ),
                    "segm_map_75": float(
                        segm_result.get("map_75", torch.tensor(0.0)).item()
                    ),
                }
            )

    return output
