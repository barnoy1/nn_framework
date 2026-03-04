from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tabulate import tabulate


def collect_class_frequency(dataset) -> tuple[Dict[int, int], Dict[int, str]]:
    if hasattr(dataset, "datasets"):
        merged_counts: Dict[int, int] = {}
        merged_names: Dict[int, str] = {}
        for child in dataset.datasets:
            child_counts, child_names = collect_class_frequency(child)
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


def _profile_single_dataset_distribution(*, dataset, split_name: str, output_dir: Path, logger) -> None:
    counts, names = collect_class_frequency(dataset)
    if not counts:
        logger.warning("Could not compute class-frequency profile for {} dataset", split_name)
        return

    total_instances = sum(counts.values())
    if total_instances <= 0:
        logger.warning("{} dataset has no instances after filtering", split_name)
        return

    rows: List[List[object]] = []
    for class_id in sorted(counts.keys()):
        class_name = names.get(class_id, str(class_id))
        frequency = int(counts[class_id])
        percentage = (100.0 * frequency) / float(total_instances)
        rows.append([class_id, class_name, frequency, f"{percentage:.2f}%"])

    table = tabulate(rows, headers=["class_id", "class_name", "frequency", "dataset_pct"], tablefmt="psql", floatfmt=".2f")
    logger.info("{} dataset class-frequency profile (instances={}):\n{}", split_name.capitalize(), total_instances, table)

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    class_ids = [row[0] for row in rows]
    frequencies = [row[2] for row in rows]
    figure_width = max(12, min(28, 0.35 * len(rows) + 8))
    plt.figure(figsize=(figure_width, 8))
    axis = sns.barplot(x=class_ids, y=frequencies, color="#9ecae1")
    axis.set_xlabel("class_id")
    axis.set_ylabel("frequency")
    axis.set_title(f"{split_name.capitalize()} class frequency distribution")
    axis.tick_params(axis="x", labelrotation=45)

    y_max = max(frequencies) if frequencies else 1
    for patch, row in zip(axis.patches, rows):
        x_pos = patch.get_x() + patch.get_width() / 2.0
        y_pos = patch.get_height()
        axis.text(x_pos, y_pos + y_max * 0.01, f"{row[1]}\n{row[3]}", ha="center", va="bottom", fontsize=8, rotation=90)

    chart_path = output_dir / f"{split_name}_class_frequency.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    if str(split_name).strip().lower() == "val":
        plt.savefig(output_dir / "labels.png", dpi=200)
    plt.close()
    logger.info("Saved class-frequency chart: {}", chart_path)


def profile_dataset_distribution(runtime, logger) -> None:
    output_dir = Path(runtime.app_config.train.output_dir) / "dataset"
    _profile_single_dataset_distribution(
        dataset=runtime.val_loader.dataset,
        split_name="val",
        output_dir=output_dir,
        logger=logger,
    )


def profile_train_and_val_dataset_distribution(runtime, logger) -> None:
    output_dir = Path(runtime.app_config.train.output_dir) / "dataset"
    _profile_single_dataset_distribution(
        dataset=runtime.train_loader.dataset,
        split_name="train",
        output_dir=output_dir,
        logger=logger,
    )
    _profile_single_dataset_distribution(
        dataset=runtime.val_loader.dataset,
        split_name="val",
        output_dir=output_dir,
        logger=logger,
    )


def dataset_num_classes(dataset) -> Optional[int]:
    if hasattr(dataset, "coco") and hasattr(dataset, "category_id_to_contiguous"):
        return len(getattr(dataset, "category_id_to_contiguous", {}))
    if hasattr(dataset, "datasets"):
        counts = [dataset_num_classes(child) for child in dataset.datasets]
        counts = [count for count in counts if count is not None]
        if not counts:
            return None
        return max(counts)
    return None


def model_num_classes(model: torch.nn.Module) -> Optional[int]:
    weight = model.state_dict().get("decoder.enc_score_head.weight")
    if weight is None or weight.ndim == 0:
        return None
    return int(weight.shape[0])
