from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from infra.common.rendering.visualize import render_prediction_with_yolo_caption
from infra.common.rendering.gt_overlay import render_ground_truth


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token or "unknown"


def _tensor_to_rgb_image(image_tensor: torch.Tensor) -> np.ndarray:
    vis_tensor = image_tensor.detach().cpu()
    if vis_tensor.ndim != 3:
        raise ValueError(
            f"Expected CHW tensor for visualization, got shape={tuple(vis_tensor.shape)}"
        )
    if int(vis_tensor.shape[0]) == 1:
        vis_tensor = vis_tensor.repeat(3, 1, 1)
    elif int(vis_tensor.shape[0]) > 3:
        vis_tensor = vis_tensor[:3]
    return (vis_tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(
        np.uint8
    )


def _save_thumbnail(rendered: np.ndarray, out_path: Path) -> None:
    rendered_image = Image.fromarray(rendered)
    rendered_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    rendered_image.save(out_path, optimize=True)


def _resolve_names(sample: Dict, resolved_epoch: int) -> tuple[Path, str, str]:
    dataset_name = safe_token(str(sample.get("dataset_name", "dataset")))
    image_id = int(sample.get("image_id", -1))
    image_stem = safe_token(Path(str(sample.get("file_name", "image"))).stem)
    folder_name = f"{dataset_name}__image_id_{image_id}"
    base_name = f"{dataset_name}__{image_stem}__epoch_{resolved_epoch:04d}"
    return Path(folder_name), folder_name, base_name


def _write_ground_truth(
    *,
    sample: Dict,
    image_folder: Path,
    base_name: str,
    class_id_to_name: Dict[int, str],
    gt_data: List[str],
) -> None:
    image_np = np.asarray(
        Image.open(Path(sample["image_path"])).convert("RGB")
    )
    rendered_gt = render_ground_truth(
        image_np,
        sample["gt_boxes"],
        sample["gt_labels"],
        class_id_to_name,
        masks=sample.get("gt_masks"),
        draw_boxes="bbox" in gt_data,
        draw_masks="masks" in gt_data,
    )
    _save_thumbnail(rendered_gt, image_folder / f"{base_name}__gt.png")


def write_sample_visualization(
    *,
    sample: Dict,
    image_tensor: torch.Tensor,
    prediction_for_vis: Dict[str, torch.Tensor],
    class_id_to_name: Dict[int, str],
    score_thr: float,
    eval_vis_dir: Path,
    resolved_epoch: int,
    vis_logger,
    step: int,
    gt_data: Optional[List[str]] = None,
) -> None:
    modes = list(gt_data or [])
    folder_rel, _, base_name = _resolve_names(sample, resolved_epoch)
    image_folder = eval_vis_dir / folder_rel
    image_folder.mkdir(parents=True, exist_ok=True)

    vis_image = _tensor_to_rgb_image(image_tensor)
    rendered = render_prediction_with_yolo_caption(
        image=vis_image,
        prediction=prediction_for_vis,
        class_id_to_name=class_id_to_name,
        confidence_threshold=score_thr,
    )
    _save_thumbnail(rendered, image_folder / f"{base_name}.png")
    vis_logger.log_image(tag="eval/visualization", image=rendered, step=step)

    if modes:
        _write_ground_truth(
            sample=sample,
            image_folder=image_folder,
            base_name=base_name,
            class_id_to_name=class_id_to_name,
            gt_data=modes,
        )
