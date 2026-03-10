from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infra.engine.flows.common.image_io import list_images, load_pil_image
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.data.preprocess import build_image_preprocess_from_loader
from infra.common.rendering.visualize import render_prediction_with_yolo_caption


def run_onnx(args, logger) -> None:
    if not args.onnx_model:
        raise ValueError("--onnx-model is required for ONNX inference")

    runtime = build_flow_runtime(
        overrides=args.overrides,
        config_path=args.config,
        build_loaders=False,
    )

    import onnxruntime as ort

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(args.onnx_model, providers=providers)
    input_name = session.get_inputs()[0].name

    transforms = build_image_preprocess_from_loader(
        runtime.app_config.data.val_dataloader, logger=logger, default_size=640
    )
    image_paths = list_images(args.input_dir)

    logger.info(
        "[mangr_inference] backend=onnx device={} images={} input={}",
        args.device,
        len(image_paths),
        args.input_dir,
    )
    if not image_paths:
        logger.warning(
            "[mangr_inference] no supported images found; nothing to process"
        )
        return

    records = []
    processed = 0
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        original_images = [load_pil_image(path) for path in batch_paths]
        batch_tensor = torch.stack(
            [transforms(image) for image in original_images], dim=0
        )
        orig_sizes = torch.tensor(
            [[image.size[0], image.size[1]] for image in original_images],
            dtype=torch.int64,
        )

        ort_outputs = session.run(
            None,
            {input_name: batch_tensor.numpy(), "orig_target_sizes": orig_sizes.numpy()},
        )
        labels_batch, boxes_batch, scores_batch = (
            ort_outputs[0],
            ort_outputs[1],
            ort_outputs[2],
        )

        for image_path, image, labels, boxes, scores in zip(
            batch_paths, original_images, labels_batch, boxes_batch, scores_batch
        ):
            rendered = render_prediction_with_yolo_caption(
                image=np.asarray(image.convert("RGB")),
                prediction={"labels": labels, "boxes": boxes, "scores": scores},
                class_id_to_name=runtime.built.class_id_to_name,
                confidence_threshold=args.score_thr,
            )
            Image.fromarray(rendered).save(output_dir / image_path.name)
            records.append(
                {
                    "image": image_path.name,
                    "labels": np.asarray(labels).tolist(),
                    "boxes": np.asarray(boxes).tolist(),
                    "scores": np.asarray(scores).tolist(),
                }
            )
            processed += 1

        logger.info("[mangr_inference] processed {}/{}", processed, len(image_paths))

    with (output_dir / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    logger.info("[mangr_inference] done. wrote {} images: {}", processed, output_dir)
