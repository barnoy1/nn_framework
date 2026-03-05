from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


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
    panels = []
    max_images = min(max(1, int(num_samples)), int(images.shape[0]))
    for index in range(max_images):
        image = images[index].detach().cpu().permute(1, 2, 0).contiguous().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        boxes = targets[index]["boxes"].detach().cpu().float().numpy()
        height, width = image.shape[0], image.shape[1]
        for cx, cy, bw, bh in boxes:
            x1 = max(0, min(width - 1, int((cx - bw * 0.5) * width)))
            y1 = max(0, min(height - 1, int((cy - bh * 0.5) * height)))
            x2 = max(0, min(width - 1, int((cx + bw * 0.5) * width)))
            y2 = max(0, min(height - 1, int((cy + bh * 0.5) * height)))
            if x2 > x1 and y2 > y1:
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

        panels.append(np.asarray(pil_image))

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
    Image.fromarray(canvas).save(output_root / f"{file_prefix}{step}{suffix}.jpg")


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
    save_eval_batch_visualization(
        output_root=output_root,
        images=images,
        targets=targets,
        step=step,
        epoch_suffix=epoch_suffix,
        num_samples=num_samples,
    )
