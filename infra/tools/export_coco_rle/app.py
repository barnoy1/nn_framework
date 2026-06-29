from __future__ import annotations

import json
import shutil
from pathlib import Path

from tabulate import tabulate

from infra.common.logging.logger import logger

from .args import parse_arguments
from .coco_reader import build_coco_from_coco_split
from .config_service import load_data_config
from .conversion_service import build_coco_for_split, save_coco_json


def invoke(args) -> None:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    conf_data_path = None
    if args.conf_data:
        conf_data_path = Path(args.conf_data).expanduser()
        if not conf_data_path.is_absolute():
            conf_data_path = (Path.cwd() / conf_data_path).resolve()
        if not conf_data_path.exists():
            raise FileNotFoundError(f"Config data file does not exist: {conf_data_path}")
    data_cfg = None
    if conf_data_path is not None:
        data_cfg = load_data_config(conf_data_path)
    has_supervisely_split = any(
        (dataset_root / split).exists()
        and not (dataset_root / split / "_annotations.coco.json").exists()
        for split in args.splits
    )
    if has_supervisely_split and data_cfg is None:
        raise ValueError("--conf_data is required for Supervisely input (ann/*.json layout)")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_conf = None
    if conf_data_path is not None:
        copied_conf = output_dir / conf_data_path.name
        shutil.copy2(conf_data_path, copied_conf)
        logger.info("Copied data config to {}", copied_conf)

    copied_experiment_conf = None
    experiment_conf_path = None
    if args.experiment_conf:
        experiment_conf_path = Path(args.experiment_conf).expanduser()
        if not experiment_conf_path.is_absolute():
            experiment_conf_path = (Path.cwd() / experiment_conf_path).resolve()
        if not experiment_conf_path.exists():
            raise FileNotFoundError(
                f"Experiment config file does not exist: {experiment_conf_path}"
            )
        copied_experiment_conf = output_dir / experiment_conf_path.name
        shutil.copy2(experiment_conf_path, copied_experiment_conf)
        logger.info("Copied experiment config to {}", copied_experiment_conf)

    remap_report = {
        "config_path": str(conf_data_path) if conf_data_path else None,
        "copied_config_path": str(copied_conf) if copied_conf else None,
        "experiment_config_path": str(experiment_conf_path)
        if experiment_conf_path
        else None,
        "copied_experiment_config_path": str(copied_experiment_conf)
        if copied_experiment_conf
        else None,
        "num_classes": data_cfg["num_classes"] if data_cfg else None,
        "mapping": (
            {str(key): value for key, value in data_cfg["mapping"].items()}
            if data_cfg
            else {}
        ),
        "splits": {},
    }

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            logger.warning("Split directory not found, skipping: {}", split_dir)
            continue

        if (split_dir / "_annotations.coco.json").exists():
            coco_data, split_remap_stats = build_coco_from_coco_split(
                split_dir, logger, data_cfg
            )
        else:
            if data_cfg is None:
                raise ValueError(
                    "--conf_data is required for Supervisely input (ann/*.json layout)"
                )
            coco_data, split_remap_stats = build_coco_for_split(
                split_dir, args.ann_subdir, args.img_subdir, logger, data_cfg
            )
        output_path = output_dir / f"instances_{split}.json"
        save_coco_json(coco_data, output_path)
        remap_report["splits"][split] = split_remap_stats

        pair_counts = split_remap_stats.get("source_target_counts", {})
        if pair_counts:
            rows = []
            for pair_key in sorted(pair_counts.keys()):
                source_contiguous_id_str, target_id_str = pair_key.split("->")
                source_label = split_remap_stats.get(
                    "source_contiguous_to_name", {}
                ).get(source_contiguous_id_str, source_contiguous_id_str)
                target_id = int(target_id_str)
                target_name_map = split_remap_stats.get("target_class_to_name", {})
                if str(target_id) in target_name_map:
                    target_label = target_name_map[str(target_id)]
                elif data_cfg:
                    target_label = data_cfg["label2classid"].get(target_id, str(target_id))
                else:
                    target_label = split_remap_stats.get(
                        "source_contiguous_to_name", {}
                    ).get(str(target_id), str(target_id))
                rows.append([source_label, target_label, int(pair_counts[pair_key])])
            logger.info(
                "Split '{}' remap summary:\n{}",
                split,
                tabulate(
                    rows,
                    headers=["orig_label", "new_label", "instances"],
                    tablefmt="psql",
                ),
            )

    remap_path = output_dir / "remap_info.json"
    with remap_path.open("w", encoding="utf-8") as file:
        json.dump(remap_report, file, indent=2)
    logger.info("Saved remap report: {}", remap_path)


def run() -> None:
    args = parse_arguments()
    invoke(args)
