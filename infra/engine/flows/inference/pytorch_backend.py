from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infra.core import to_result_list
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.engine.flows.common.image_io import list_images, load_pil_image
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.utils.viz.visualize import render_prediction_with_yolo_caption


def run_pytorch(args, logger) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for PyTorch inference")

    runtime = build_flow_runtime(
        overrides=args.overrides,
        config_path=args.config,
        build_loaders=False,
    )
    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    model = runtime.built.model
    postprocessor = runtime.built.postprocessor
    class_id_to_name = runtime.built.class_id_to_name

    if hasattr(model, "deploy"):
        model = model.deploy()
    if hasattr(postprocessor, "deploy"):
        postprocessor = postprocessor.deploy()

    device = torch.device(args.device)
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_image_preprocess_from_loader(runtime.app_config.data.val_dataloader, logger=logger, default_size=640)
    image_paths = list_images(args.input_dir)

    logger.info("[mangr_inference] backend=pytorch device={} images={} input={}", args.device, len(image_paths), args.input_dir)
    if not image_paths:
        logger.warning("[mangr_inference] no supported images found; nothing to process")
        return

    records = []
    processed = 0
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        original_images = [load_pil_image(path) for path in batch_paths]
        batch_tensor = torch.stack([transforms(image) for image in original_images], dim=0).to(device)
        orig_sizes = torch.tensor([[image.size[0], image.size[1]] for image in original_images], device=device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            results = to_result_list(outputs, postprocessor, orig_sizes)

        for image_path, image, result in zip(batch_paths, original_images, results):
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()

            rendered = render_prediction_with_yolo_caption(
                image=np.asarray(image.convert("RGB")),
                prediction=result,
                class_id_to_name=class_id_to_name,
                confidence_threshold=args.score_thr,
            )
            Image.fromarray(rendered).save(output_dir / image_path.name)

            records.append({"image": image_path.name, "labels": labels.tolist(), "boxes": boxes.tolist(), "scores": scores.tolist()})
            processed += 1

        logger.info("[mangr_inference] processed {}/{}", processed, len(image_paths))

    with (output_dir / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    logger.info("[mangr_inference] done. wrote {} images + {}", processed, output_dir / "detections.json")
