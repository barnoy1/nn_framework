from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .visualize import (
    _apply_transparent_bbox_fill,
    _as_numpy_array,
    _build_detection_labels,
)


def render_ground_truth(
    image: np.ndarray,
    boxes: Any,
    labels: Any,
    class_id_to_name: Optional[Dict[int, str]] = None,
    *,
    masks: Any = None,
    draw_boxes: bool = True,
    draw_masks: bool = False,
) -> np.ndarray:
    import supervision as sv

    boxes_array = _as_numpy_array(boxes).astype(float).reshape(-1, 4)
    labels_array = _as_numpy_array(labels).astype(int).reshape(-1)
    masks_array = None
    if draw_masks and masks is not None:
        masks_array = _as_numpy_array(masks).astype(bool)
        if masks_array.shape[0] == 0:
            masks_array = None

    detections = sv.Detections(
        xyxy=boxes_array,
        class_id=labels_array,
        mask=masks_array,
    )

    frame = image.copy()
    if draw_masks and detections.mask is not None:
        frame = sv.MaskAnnotator().annotate(frame, detections)
    if draw_boxes:
        frame = _apply_transparent_bbox_fill(frame, detections)
        frame = sv.BoxAnnotator().annotate(frame, detections)
        labels_text = _build_detection_labels(detections, class_id_to_name)
        frame = sv.LabelAnnotator().annotate(frame, detections, labels_text)
    return frame


def _demo() -> None:
    # ponytail: self-check for the box/mask render branches; no sv import needed offline.
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    boxes = np.array([[2.0, 2.0, 10.0, 10.0]])
    labels = np.array([0])
    masks = np.zeros((1, 20, 20), dtype=bool)
    masks[0, 2:10, 2:10] = True
    for draw_boxes, draw_masks in ((True, False), (False, True), (True, True)):
        out = render_ground_truth(
            image,
            boxes,
            labels,
            {0: "obj"},
            masks=masks,
            draw_boxes=draw_boxes,
            draw_masks=draw_masks,
        )
        assert out.shape == image.shape, out.shape
    print("render_ground_truth demo ok")


if __name__ == "__main__":
    _demo()
