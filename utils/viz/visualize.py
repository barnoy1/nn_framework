from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np


def rtdetr_output_to_sv_detections(
    prediction: Dict,
    confidence_threshold: float = 0.3,
) -> Any:
    import supervision as sv

    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy().astype(int)

    keep = scores >= confidence_threshold
    masks = None
    if "masks" in prediction and prediction["masks"] is not None:
        masks = prediction["masks"].detach().cpu().numpy().astype(bool)
        masks = masks[keep]

    return sv.Detections(
        xyxy=boxes[keep],
        confidence=scores[keep],
        class_id=labels[keep],
        mask=masks,
    )


def save_side_by_side(
    image: np.ndarray,
    gt: Any,
    pred: Any,
    out_path: Path,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    import supervision as sv

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    gt_frame = image.copy()
    pred_frame = image.copy()

    if gt.mask is not None:
        gt_frame = mask_annotator.annotate(gt_frame, gt)
    gt_frame = box_annotator.annotate(gt_frame, gt)

    if pred.mask is not None:
        pred_frame = mask_annotator.annotate(pred_frame, pred)
    pred_frame = box_annotator.annotate(pred_frame, pred)

    combined = np.concatenate([gt_frame, pred_frame], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sv.cv2.imwrite(str(out_path), combined)
