from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

import cv2
import numpy as np


DEFAULT_BBOX_ALPHA = 0.2
DEFAULT_BBOX_FILL_COLOR = (0, 255, 0)


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
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    gt_frame = image.copy()
    pred_frame = image.copy()
    gt_labels = _build_detection_labels(gt, class_names)
    pred_labels = _build_detection_labels(pred, class_names)

    if gt.mask is not None:
        gt_frame = mask_annotator.annotate(gt_frame, gt)
    gt_frame = _apply_transparent_bbox_fill(gt_frame, gt)
    gt_frame = box_annotator.annotate(gt_frame, gt)
    gt_frame = label_annotator.annotate(gt_frame, gt, gt_labels)

    if pred.mask is not None:
        pred_frame = mask_annotator.annotate(pred_frame, pred)
    pred_frame = _apply_transparent_bbox_fill(pred_frame, pred)
    pred_frame = box_annotator.annotate(pred_frame, pred)
    pred_frame = label_annotator.annotate(pred_frame, pred, pred_labels)

    combined = np.concatenate([gt_frame, pred_frame], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sv.cv2.imwrite(str(out_path), combined)


def _label_name(class_id: int, class_id_to_name: Optional[Dict[int, str]]) -> str:
    if class_id_to_name is None:
        return str(class_id)
    return class_id_to_name.get(int(class_id), str(class_id))


def _as_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _build_sv_detections(
    *,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
):
    import supervision as sv

    scores_array = _as_numpy_array(scores).astype(float)
    boxes_array = _as_numpy_array(boxes).astype(float)
    labels_array = _as_numpy_array(labels).astype(int)

    keep = scores_array >= float(confidence_threshold)
    return sv.Detections(
        xyxy=boxes_array[keep],
        confidence=scores_array[keep],
        class_id=labels_array[keep],
    )


def _build_detection_labels(
    detections: Any,
    class_id_to_name: Optional[Dict[int, str]],
) -> list[str]:
    class_ids = getattr(detections, "class_id", None)
    if class_ids is None:
        return []

    confidences = getattr(detections, "confidence", None)
    labels: list[str] = []
    for index, class_id in enumerate(class_ids):
        label = _label_name(int(class_id), class_id_to_name)
        if confidences is None:
            labels.append(label)
            continue
        labels.append(f"{label} {float(confidences[index]):.2f}")
    return labels


def _apply_transparent_bbox_fill(
    frame: np.ndarray,
    detections: Any,
    *,
    alpha: float = DEFAULT_BBOX_ALPHA,
    fill_color: tuple[int, int, int] = DEFAULT_BBOX_FILL_COLOR,
) -> np.ndarray:
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None or len(xyxy) == 0 or alpha <= 0.0:
        return frame

    overlay = frame.copy()
    height, width = frame.shape[:2]
    for box in xyxy:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, thickness=cv2.FILLED)

    return cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0.0)


def draw_yolo_caption_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_id_to_name: Optional[Dict[int, str]] = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    import supervision as sv

    detections = _build_sv_detections(
        boxes=boxes,
        labels=labels,
        scores=scores,
        confidence_threshold=confidence_threshold,
    )
    captions = _build_detection_labels(detections, class_id_to_name)

    frame = image.copy()
    frame = _apply_transparent_bbox_fill(frame, detections)
    frame = sv.BoxAnnotator().annotate(frame, detections)
    return sv.LabelAnnotator().annotate(frame, detections, captions)


def render_prediction_with_yolo_caption(
    image: np.ndarray,
    prediction: Dict,
    class_id_to_name: Optional[Dict[int, str]] = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    return draw_yolo_caption_detections(
        image=image,
        boxes=_as_numpy_array(prediction["boxes"]),
        labels=_as_numpy_array(prediction["labels"]),
        scores=_as_numpy_array(prediction["scores"]),
        class_id_to_name=class_id_to_name,
        confidence_threshold=confidence_threshold,
    )


__all__ = [
    "draw_yolo_caption_detections",
    "render_prediction_with_yolo_caption",
    "rtdetr_output_to_sv_detections",
    "save_side_by_side",
]
