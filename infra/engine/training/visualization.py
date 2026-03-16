from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image


def _to_rgb_uint8(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels > 3:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape for visualization: {image.shape}")

    return image


def _target_to_sv_detections(target, image: np.ndarray) -> tuple[sv.Detections, list[str]]:
    boxes = target["boxes"].detach().cpu().float().numpy()
    labels_tensor = target.get("labels")
    labels = (
        labels_tensor.detach().cpu().numpy().astype(int)
        if isinstance(labels_tensor, torch.Tensor)
        else None
    )
    height, width = image.shape[0], image.shape[1]

    xyxy_boxes = []
    caption_labels: list[str] = []
    for index, (cx, cy, bw, bh) in enumerate(boxes):
        x1 = max(0, min(width - 1, int((cx - bw * 0.5) * width)))
        y1 = max(0, min(height - 1, int((cy - bh * 0.5) * height)))
        x2 = max(0, min(width - 1, int((cx + bw * 0.5) * width)))
        y2 = max(0, min(height - 1, int((cy + bh * 0.5) * height)))
        if x2 <= x1 or y2 <= y1:
            continue
        xyxy_boxes.append([x1, y1, x2, y2])
        if labels is not None:
            caption_labels.append(str(labels[index]))

    if not xyxy_boxes:
        return sv.Detections.empty(), []

    detections = sv.Detections(
        xyxy=np.asarray(xyxy_boxes, dtype=np.float32),
        class_id=(
            np.asarray([int(label) for label in caption_labels], dtype=int)
            if caption_labels
            else None
        ),
    )
    return detections, caption_labels


def _save_batch_visualization(
    *,
    output_root: Path,
    images: torch.Tensor,
    targets,
    step: int,
    file_prefix: str,
    epoch_suffix: int | None = None,
    num_samples: int = 4,
) -> None:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    panels = []
    max_images = min(max(1, int(num_samples)), int(images.shape[0]))
    for index in range(max_images):
        image = _to_rgb_uint8(images[index])
        detections, labels = _target_to_sv_detections(targets[index], image)
        panel = box_annotator.annotate(scene=image.copy(), detections=detections)
        if labels:
            panel = label_annotator.annotate(
                scene=panel,
                detections=detections,
                labels=labels,
            )
        panels.append(panel)

    if not panels:
        return

    cols = 2
    rows = int(math.ceil(len(panels) / float(cols)))
    panel_h, panel_w = panels[0].shape[0], panels[0].shape[1]
    canvas = np.zeros((rows * panel_h, cols * panel_w, 3), dtype=np.uint8)
    for panel_index, panel in enumerate(panels):
        row = panel_index // cols
        col = panel_index % cols
        y0 = row * panel_h
        x0 = col * panel_w
        canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel

    suffix = ""
    if epoch_suffix is not None:
        suffix = f"_{int(epoch_suffix):04d}"
    output_image = Image.fromarray(canvas)
    output_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    output_image.save(
        output_root / f"{file_prefix}{step}{suffix}.jpg", quality=85, optimize=True
    )


def save_train_batch_visualization(
    output_root: Path,
    images: torch.Tensor,
    targets,
    step: int,
    num_samples: int = 4,
) -> None:
    _save_batch_visualization(
        output_root=output_root,
        images=images,
        targets=targets,
        step=step,
        file_prefix="train_batch",
        epoch_suffix=None,
        num_samples=num_samples,
    )


def save_eval_batch_visualization(
    output_root: Path,
    images: torch.Tensor,
    targets,
    step: int,
    epoch_suffix: int | None = None,
    num_samples: int = 4,
) -> None:
    _save_batch_visualization(
        output_root=output_root,
        images=images,
        targets=targets,
        step=step,
        file_prefix="eval_batch",
        epoch_suffix=epoch_suffix,
        num_samples=num_samples,
    )


def save_val_batch_visualization(
    output_root: Path,
    images: torch.Tensor,
    targets,
    step: int,
    epoch_suffix: int | None = None,
    num_samples: int = 4,
) -> None:
    _save_batch_visualization(
        output_root=output_root,
        images=images,
        targets=targets,
        step=step,
        file_prefix="val_batch",
        epoch_suffix=epoch_suffix,
        num_samples=num_samples,
    )
