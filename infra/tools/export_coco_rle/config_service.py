from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yaml


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_data_config(conf_data_path: Path) -> dict:
    with conf_data_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    data_cfg = payload.get("data", payload if isinstance(payload, dict) else {})
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Invalid data config structure in {conf_data_path}")

    label_map_raw = data_cfg.get("label2classid", {})
    if not isinstance(label_map_raw, dict) or not label_map_raw:
        raise ValueError("data.label2classid must be a non-empty mapping")

    label2classid: Dict[int, str] = {}
    for key, value in label_map_raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid class id in label2classid: {key}") from error
        class_name = str(value).strip()
        if not class_name:
            raise ValueError(
                f"Empty class name for class id {class_id} in label2classid"
            )
        label2classid[class_id] = class_name

    mapping_raw = data_cfg.get("mapping", {})
    if mapping_raw is None:
        mapping_raw = {}
    if not isinstance(mapping_raw, dict):
        raise ValueError(
            "data.mapping must be a mapping of source_class_id -> target_class_id"
        )

    mapping: Dict[int, int] = {}
    for source_key, target_value in mapping_raw.items():
        try:
            source_id = int(source_key)
            target_id = int(target_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid mapping entry: {source_key}: {target_value}"
            ) from error
        mapping[source_id] = target_id

    num_classes = int(data_cfg.get("num_classes", len(label2classid)))
    if num_classes <= 0:
        raise ValueError("data.num_classes must be > 0")

    for source_id, target_id in mapping.items():
        if target_id < 0 or target_id >= num_classes:
            raise ValueError(
                f"Mapped target class id out of range for data.num_classes={num_classes}: {source_id} -> {target_id}"
            )

    return {
        "label2classid": label2classid,
        "mapping": mapping,
        "num_classes": num_classes,
    }
