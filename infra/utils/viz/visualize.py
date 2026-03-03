from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any
import colorsys

import numpy as np
import cv2


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


def _label_name(class_id: int, class_id_to_name: Optional[Dict[int, str]]) -> str:
    if class_id_to_name is None:
        return str(class_id)
    return class_id_to_name.get(int(class_id), str(class_id))


def _pastel_bgr_for_class(class_id: int) -> tuple[int, int, int]:
    class_index = int(class_id) % 360
    hue = (class_index * 0.618033988749895) % 1.0
    saturation = 0.45
    value = 0.95
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(blue * 255), int(green * 255), int(red * 255)


def _readable_text_color_for_bgr(color_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color_bgr
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (20, 20, 20) if luminance > 170 else (245, 245, 245)


def draw_yolo_caption_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_id_to_name: Optional[Dict[int, str]] = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    frame = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < confidence_threshold:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = _label_name(int(label), class_id_to_name)
        caption = f"{class_name} {float(score):.1f}"
        class_color = _pastel_bgr_for_class(int(label))
        text_color = _readable_text_color_for_bgr(class_color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_top = max(0, y1 - text_h - baseline - 6)
        box_bottom = max(text_h + baseline + 6, y1)
        box_right = min(frame.shape[1] - 1, x1 + text_w + 6)

        cv2.rectangle(frame, (x1, box_top), (box_right, box_bottom), class_color, -1)
        cv2.putText(
            frame,
            caption,
            (x1 + 3, box_bottom - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def render_prediction_with_yolo_caption(
    image: np.ndarray,
    prediction: Dict,
    class_id_to_name: Optional[Dict[int, str]] = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    labels = prediction["labels"]
    boxes = prediction["boxes"]
    scores = prediction["scores"]

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()

    return draw_yolo_caption_detections(
        image=image,
        boxes=np.asarray(boxes),
        labels=np.asarray(labels),
        scores=np.asarray(scores),
        class_id_to_name=class_id_to_name,
        confidence_threshold=confidence_threshold,
    )
