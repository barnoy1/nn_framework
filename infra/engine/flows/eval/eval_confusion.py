from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torchvision

ConfusionEvent = Tuple[Optional[int], Optional[int]]

IOU_MATCH_THRESHOLD = 0.5


def build_confusion_events(
    *,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> List[ConfusionEvent]:
    pred_labels = pred_labels.detach().cpu().long()
    gt_labels = gt_labels.detach().cpu().long()
    events: List[ConfusionEvent] = []

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        events.extend((int(gt_labels[i].item()), None) for i in range(gt_labels.numel()))
        events.extend((None, int(pred_labels[i].item())) for i in range(pred_labels.numel()))
        return events

    ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
    candidate_pairs = [
        (float(ious[pred_idx, gt_idx].item()), pred_idx, gt_idx)
        for pred_idx in range(ious.shape[0])
        for gt_idx in range(ious.shape[1])
        if float(ious[pred_idx, gt_idx].item()) >= IOU_MATCH_THRESHOLD
    ]
    candidate_pairs.sort(key=lambda item: item[0], reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for _, pred_idx, gt_idx in candidate_pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        events.append(
            (int(gt_labels[gt_idx].item()), int(pred_labels[pred_idx].item()))
        )

    events.extend(
        (int(gt_labels[i].item()), None)
        for i in range(gt_labels.numel())
        if i not in matched_gt
    )
    events.extend(
        (None, int(pred_labels[i].item()))
        for i in range(pred_labels.numel())
        if i not in matched_pred
    )
    return events
