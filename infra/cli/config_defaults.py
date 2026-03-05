from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import ACTION_TO_RUNTIME_SECTION, REPO_ROOT


def _resolve_relocated_experiment_path(candidate: Path) -> Path | None:
    parts = candidate.parts
    marker = ("infra", "config", "hydra", "experiment")
    if len(parts) < len(marker):
        return None
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            file_name = candidate.name
            relocated = (REPO_ROOT / "experiment" / file_name).resolve()
            if relocated.exists():
                return relocated
            return None
    return None


def resolve_config_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        relocated = _resolve_relocated_experiment_path(candidate)
        if relocated is not None:
            return relocated
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate


def load_config_payload(path: str) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def build_parser_defaults(config_path: str, action: str) -> dict[str, Any]:
    payload = load_config_payload(config_path)
    runtime_cfg = payload.get("runtime", {}) if isinstance(payload.get("runtime", {}), dict) else {}
    runtime_common = runtime_cfg.get("common", {}) if isinstance(runtime_cfg.get("common", {}), dict) else {}
    runtime_data_prep = (
        runtime_cfg.get("data_preparation", {})
        if isinstance(runtime_cfg.get("data_preparation", {}), dict)
        else {}
    )
    runtime_actions = runtime_cfg.get("actions", {}) if isinstance(runtime_cfg.get("actions", {}), dict) else {}
    runtime_action_key = ACTION_TO_RUNTIME_SECTION.get(action, action)
    runtime_action = (
        runtime_actions.get(runtime_action_key, {})
        if isinstance(runtime_actions.get(runtime_action_key, {}), dict)
        else {}
    )
    train_cfg = payload.get("train", {}) if isinstance(payload.get("train", {}), dict) else {}
    data_cfg = payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {}

    device_default = str(runtime_common.get("device", "cuda"))
    if "device" not in runtime_common and not bool(runtime_common.get("use_gpu", True)):
        device_default = "cpu"
    splits_default = runtime_data_prep.get("supervisely_splits", ["train", "valid"])
    if not isinstance(splits_default, list):
        splits_default = ["train", "valid"]

    defaults = {
        "config": str(resolve_config_path(config_path)),
        "output_dir": str(runtime_common.get("output_dir", str(REPO_ROOT / "out"))),
        "device": device_default,
        "batch_size": int(runtime_common.get("batch_size", train_cfg.get("batch_size", 1))),
        "num_workers": int(runtime_common.get("num_workers", train_cfg.get("num_workers", 2))),
        "checkpoint": str(runtime_common.get("checkpoint", "")),
        "input_dir": "",
        "onnx_model": "",
        "dataset_conf": str(resolve_config_path(config_path)),
        "experiment_conf": None,
        "dataset_root": data_cfg.get("dataset_root"),
        "ann_subdir": str(runtime_data_prep.get("ann_subdir", "ann")),
        "img_subdir": str(runtime_data_prep.get("img_subdir", "img")),
        "splits": splits_default,
    }

    for key in (
        "output_dir",
        "device",
        "batch_size",
        "num_workers",
        "checkpoint",
        "input_dir",
        "onnx_model",
        "dataset_conf",
        "experiment_conf",
        "dataset_root",
        "ann_subdir",
        "img_subdir",
        "splits",
    ):
        if key not in runtime_action:
            continue
        value = runtime_action[key]
        if key in ("batch_size", "num_workers"):
            defaults[key] = int(value)
        elif key == "splits":
            defaults[key] = value if isinstance(value, list) else [str(value)]
        elif key in ("dataset_conf", "experiment_conf", "config") and value:
            defaults[key] = str(resolve_config_path(str(value)))
        else:
            defaults[key] = value

    if action == "train":
        train_checkpoint = str(defaults.get("checkpoint") or "").strip()
        if not train_checkpoint:
            eval_cfg = runtime_actions.get("eval", {}) if isinstance(runtime_actions.get("eval", {}), dict) else {}
            eval_checkpoint = str(eval_cfg.get("checkpoint") or "").strip()
            if eval_checkpoint:
                defaults["checkpoint"] = eval_checkpoint

    return defaults


def load_dataset_export_settings(dataset_conf: str | None) -> tuple[list[str], str, str]:
    splits = ["train", "valid"]
    ann_subdir = "ann"
    img_subdir = "img"

    if not dataset_conf:
        return splits, ann_subdir, img_subdir

    conf_path = Path(dataset_conf).expanduser()
    if not conf_path.is_absolute():
        conf_path = (REPO_ROOT / conf_path).resolve()
    if not conf_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    data_cfg = payload.get("data", payload if isinstance(payload, dict) else {})
    if not isinstance(data_cfg, dict):
        return splits, ann_subdir, img_subdir

    if isinstance(data_cfg.get("ann_subdir"), str) and data_cfg.get("ann_subdir"):
        ann_subdir = data_cfg["ann_subdir"]
    if isinstance(data_cfg.get("img_subdir"), str) and data_cfg.get("img_subdir"):
        img_subdir = data_cfg["img_subdir"]

    discovered_splits: list[str] = []
    for loader_key in ("train_dataloader", "val_dataloader"):
        loader_cfg = data_cfg.get(loader_key)
        if not isinstance(loader_cfg, dict):
            continue
        dataset_cfg = loader_cfg.get("dataset")
        if not isinstance(dataset_cfg, dict):
            continue
        datasets = dataset_cfg.get("datasets")
        if not isinstance(datasets, list):
            continue
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            img_folder = entry.get("img_folder") or entry.get("img_dir")
            if not isinstance(img_folder, str) or not img_folder:
                continue
            split_name = Path(img_folder).parent.name
            if split_name and split_name not in discovered_splits:
                discovered_splits.append(split_name)

    if discovered_splits:
        splits = discovered_splits

    return splits, ann_subdir, img_subdir


def resolve_experiment_conf_path(dataset_conf: str) -> str | None:
    conf_path = Path(dataset_conf).expanduser()
    if not conf_path.is_absolute():
        conf_path = (REPO_ROOT / conf_path).resolve()

    if "experiment" in conf_path.parts and conf_path.exists():
        return str(conf_path)

    hydra_root = REPO_ROOT / "infra" / "config" / "hydra"
    root_config = hydra_root / "config.yaml"
    if not root_config.exists():
        return None

    with root_config.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    defaults = payload.get("defaults", [])
    if not isinstance(defaults, list):
        return None

    experiment_name = None
    for entry in defaults:
        if isinstance(entry, dict) and "experiment" in entry:
            experiment_name = entry.get("experiment")
            break

    if not isinstance(experiment_name, str) or not experiment_name:
        return None

    experiment_path = REPO_ROOT / "experiment" / f"{experiment_name}.yaml"
    if experiment_path.exists():
        return str(experiment_path.resolve())

    experiment_path = hydra_root / "experiment" / f"{experiment_name}.yaml"
    if experiment_path.exists():
        return str(experiment_path.resolve())
    return None
