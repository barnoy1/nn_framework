from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infra.core import to_canonical_predictions
from infra.common.rendering.visualize import render_prediction_with_yolo_caption


@torch.no_grad()
def save_eval_visualizations(runtime, args, logger) -> None:
    vis_dir = Path(runtime.app_config.engine.execution.output_dir) / "inference" / "eval"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = runtime.built.model
    postprocessor = runtime.built.postprocessor
    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    device = torch.device(args.device)
    model.to(device).eval()

    saved = 0
    for images, targets in runtime.val_loader:
        images = images.to(device, non_blocking=True)
        targets = [
            {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in target.items()
            }
            for target in targets
        ]

        outputs = model(images)
        orig_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results = to_canonical_predictions(
            outputs, postprocessor, orig_sizes, iou_types=runtime.app_config.engine.data.iou_types
        )

        for image_tensor, target, prediction in zip(images, targets, results):
            file_path = target.get("file_path")
            if isinstance(file_path, str) and file_path:
                image_np = np.asarray(Image.open(file_path).convert("RGB"))
            else:
                image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
                image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            rendered = render_prediction_with_yolo_caption(
                image=image_np,
                prediction=prediction,
                class_id_to_name=runtime.built.class_id_to_name,
                confidence_threshold=args.score_thr,
            )
            Image.fromarray(rendered).save(vis_dir / f"eval_{saved:05d}.jpg")
            saved += 1
            if saved >= args.vis_samples:
                logger.info("Saved {} eval visualizations to {}", saved, vis_dir)
                return

    logger.info("Saved {} eval visualizations to {}", saved, vis_dir)
