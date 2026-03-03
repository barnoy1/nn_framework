from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nn_framework.flows.common.runtime import build_flow_runtime
from nn_framework.engine.callbacks import CallbackList
from nn_framework.engine.trainer import Trainer
from nn_framework.utils.log import logger
from nn_framework.utils.viz.visualize import render_prediction_with_yolo_caption


def _collect_class_frequency(dataset) -> tuple[Dict[int, int], Dict[int, str]]:
    if hasattr(dataset, "datasets"):
        merged_counts: Dict[int, int] = {}
        merged_names: Dict[int, str] = {}
        for child in dataset.datasets:
            child_counts, child_names = _collect_class_frequency(child)
            for class_id, count in child_counts.items():
                merged_counts[class_id] = merged_counts.get(class_id, 0) + int(count)
            merged_names.update(child_names)
        return merged_counts, merged_names

    if not hasattr(dataset, "coco") or not hasattr(dataset, "category_id_to_contiguous"):
        return {}, {}

    coco = dataset.coco
    category_map = dataset.category_id_to_contiguous
    contiguous_to_name: Dict[int, str] = {}

    for category in coco.loadCats(coco.getCatIds()):
        category_id = category.get("id")
        if category_id in category_map:
            contiguous_id = int(category_map[category_id])
            contiguous_to_name[contiguous_id] = str(category.get("name", contiguous_id))

    frequencies: Dict[int, int] = {class_id: 0 for class_id in contiguous_to_name.keys()}
    for annotation in coco.anns.values():
        if annotation.get("iscrowd", 0) == 1:
            continue
        contiguous_id = category_map.get(annotation.get("category_id"))
        if contiguous_id is None:
            continue
        class_id = int(contiguous_id)
        frequencies[class_id] = frequencies.get(class_id, 0) + 1

    return frequencies, contiguous_to_name


def _profile_dataset_distribution(runtime) -> None:
    counts, names = _collect_class_frequency(runtime.val_loader.dataset)
    if not counts:
        logger.warning("Could not compute class-frequency profile for validation dataset")
        return

    total_instances = sum(counts.values())
    if total_instances <= 0:
        logger.warning("Validation dataset has no instances after filtering")
        return

    rows: List[List[object]] = []
    for class_id in sorted(counts.keys()):
        class_name = names.get(class_id, str(class_id))
        frequency = int(counts[class_id])
        percentage = (100.0 * frequency) / float(total_instances)
        rows.append([class_id, class_name, frequency, f"{percentage:.2f}%"])

    from tabulate import tabulate

    table = tabulate(
        rows,
        headers=["class_id", "class_name", "frequency", "dataset_pct"],
        tablefmt="psql",
        floatfmt=".2f",
    )
    logger.info("Validation dataset class-frequency profile (instances={}):\n{}", total_instances, table)

    output_dir = Path(runtime.app_config.train.output_dir) / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    class_ids = [row[0] for row in rows]
    frequencies = [row[2] for row in rows]

    figure_width = max(12, min(28, 0.35 * len(rows) + 8))
    plt.figure(figsize=(figure_width, 8))
    axis = sns.barplot(x=class_ids, y=frequencies, color="#9ecae1")

    axis.set_xlabel("class_id")
    axis.set_ylabel("frequency")
    axis.set_title("Validation class frequency distribution")
    axis.tick_params(axis="x", labelrotation=45)

    y_max = max(frequencies) if frequencies else 1
    for patch, row in zip(axis.patches, rows):
        x_pos = patch.get_x() + patch.get_width() / 2.0
        y_pos = patch.get_height()
        label = f"{row[1]}\n{row[3]}"
        axis.text(x_pos, y_pos + y_max * 0.01, label, ha="center", va="bottom", fontsize=8, rotation=90)

    chart_path = output_dir / "val_class_frequency.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    logger.info("Saved class-frequency chart: {}", chart_path)


def _dataset_num_classes(dataset) -> Optional[int]:
    if hasattr(dataset, "coco") and hasattr(dataset, "category_id_to_contiguous"):
        return len(getattr(dataset, "category_id_to_contiguous", {}))

    if hasattr(dataset, "datasets"):
        counts = [_dataset_num_classes(child) for child in dataset.datasets]
        counts = [count for count in counts if count is not None]
        if not counts:
            return None
        return max(counts)

    return None


def _model_num_classes(model: torch.nn.Module) -> Optional[int]:
    weight = model.state_dict().get("decoder.enc_score_head.weight")
    if weight is None or weight.ndim == 0:
        return None
    return int(weight.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework evaluation manager")
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-class-mismatch", action="store_true")
    parser.add_argument("--vis-samples", type=int, default=16)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def _to_result_list(outputs, postprocessor, orig_sizes):
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs

    if isinstance(outputs, dict):
        processed = postprocessor(outputs, orig_sizes)
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        labels, boxes, scores = outputs
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]
    else:
        processed = postprocessor(outputs, orig_sizes)

    if isinstance(processed, list):
        return processed

    if isinstance(processed, (tuple, list)) and len(processed) == 3:
        labels, boxes, scores = processed
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    if isinstance(processed, dict) and {"labels", "boxes", "scores"}.issubset(set(processed.keys())):
        labels = processed["labels"]
        boxes = processed["boxes"]
        scores = processed["scores"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    raise TypeError(f"Unsupported eval output format: {type(processed)}")


@torch.no_grad()
def _save_eval_visualizations(runtime, args: argparse.Namespace) -> None:
    vis_dir = Path(runtime.app_config.train.output_dir) / "inference" / "eval"
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
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]

        outputs = model(images)
        orig_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results = _to_result_list(outputs, postprocessor, orig_sizes)

        for image_tensor, prediction in zip(images, results):
            image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
            image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

            rendered = render_prediction_with_yolo_caption(
                image=image_np,
                prediction=prediction,
                class_id_to_name=runtime.built.class_id_to_name,
                confidence_threshold=args.score_thr,
            )

            out_path = vis_dir / f"eval_{saved:05d}.jpg"
            Image.fromarray(rendered).save(out_path)
            saved += 1
            if saved >= args.vis_samples:
                logger.info("Saved {} eval visualizations to {}", saved, vis_dir)
                return

    logger.info("Saved {} eval visualizations to {}", saved, vis_dir)


def main() -> None:
    args = parse_args()
    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides)

    _profile_dataset_distribution(runtime)

    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(
        runtime.built.model,
        state,
        allow_mismatch=args.allow_class_mismatch,
    )
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    dataset_classes = _dataset_num_classes(runtime.val_loader.dataset)
    model_classes = _model_num_classes(runtime.built.model)
    if dataset_classes is not None and model_classes is not None and dataset_classes != model_classes:
        logger.warning(
            "validation dataset classes ({}) differ from model classes ({}). "
            "Evaluation AP can collapse to zero due to class-id mismatch.",
            dataset_classes,
            model_classes,
        )

    _save_eval_visualizations(runtime, args)

    trainer = Trainer(
        app_config=runtime.app_config,
        model=runtime.built.model,
        criterion=runtime.built.criterion,
        postprocessor=runtime.built.postprocessor,
        optimizer=runtime.built.optimizer,
        scheduler=runtime.built.scheduler,
        train_loader=runtime.train_loader,
        val_loader=runtime.val_loader,
        callbacks=CallbackList([]),
        ema_model=runtime.built.ema_model,
        model_wrapper=runtime.wrapper,
    )

    metrics = trainer.validate(epoch=0)
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("{}: {}", key, value)


if __name__ == "__main__":
    main()
