from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List

import yaml

from infra.utils.log import logger, setup_logger

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "raw_models" / "RT-DETR" / "rtdetrv2_pytorch"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ActionHandler = Callable[[argparse.Namespace], None]

_ACTION_TO_RUNTIME_SECTION = {
    "train": "train",
    "eval": "eval",
    "inference": "inference",
    "inference-onnx": "inference_onnx",
    "export-onnx": "export_onnx",
    "export-coco-rle": "export_coco_rle",
}


def _resolve_config_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate


def _load_config_payload(path: str) -> dict[str, Any]:
    config_path = _resolve_config_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _build_parser_defaults(config_path: str, action: str) -> dict[str, Any]:
    payload = _load_config_payload(config_path)
    runtime_cfg = payload.get("runtime", {}) if isinstance(payload.get("runtime", {}), dict) else {}
    runtime_common = runtime_cfg.get("common", {}) if isinstance(runtime_cfg.get("common", {}), dict) else {}
    runtime_data_prep = (
        runtime_cfg.get("data_preparation", {})
        if isinstance(runtime_cfg.get("data_preparation", {}), dict)
        else {}
    )
    runtime_actions = runtime_cfg.get("actions", {}) if isinstance(runtime_cfg.get("actions", {}), dict) else {}
    runtime_action_key = _ACTION_TO_RUNTIME_SECTION.get(action, action)
    runtime_action = (
        runtime_actions.get(runtime_action_key, {})
        if isinstance(runtime_actions.get(runtime_action_key, {}), dict)
        else {}
    )
    train_cfg = payload.get("train", {}) if isinstance(payload.get("train", {}), dict) else {}
    model_cfg = payload.get("model", {}) if isinstance(payload.get("model", {}), dict) else {}
    data_cfg = payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {}

    device_default = str(runtime_common.get("device", "cuda"))
    if "device" not in runtime_common and not bool(runtime_common.get("use_gpu", True)):
        device_default = "cpu"
    splits_default = runtime_data_prep.get("supervisely_splits", ["train", "valid"])
    if not isinstance(splits_default, list):
        splits_default = ["train", "valid"]

    defaults = {
        "config": str(_resolve_config_path(config_path)),
        "model_profile": str(model_cfg.get("variant", "r18")),
        "output_dir": str(runtime_common.get("output_dir", str(REPO_ROOT / "out"))),
        "device": device_default,
        "batch_size": int(runtime_common.get("batch_size", train_cfg.get("batch_size", 1))),
        "num_workers": int(runtime_common.get("num_workers", train_cfg.get("num_workers", 2))),
        "checkpoint": "",
        "input_dir": "",
        "onnx_model": "",
        "dataset_conf": str(_resolve_config_path(config_path)),
        "experiment_conf": None,
        "dataset_root": data_cfg.get("dataset_root"),
        "ann_subdir": str(runtime_data_prep.get("ann_subdir", "ann")),
        "img_subdir": str(runtime_data_prep.get("img_subdir", "img")),
        "splits": splits_default,
    }

    if "model_profile" in runtime_common:
        defaults["model_profile"] = str(runtime_common["model_profile"])

    for key in (
        "output_dir",
        "model_profile",
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
            defaults[key] = str(_resolve_config_path(str(value)))
        else:
            defaults[key] = value

    return defaults


def _load_dataset_export_settings(dataset_conf: str | None) -> tuple[list[str], str, str]:
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


def _resolve_experiment_conf_path(dataset_conf: str) -> str | None:
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

    experiment_path = hydra_root / "experiment" / f"{experiment_name}.yaml"
    if experiment_path.exists():
        return str(experiment_path.resolve())
    return None


def _resolve_checkpoint_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    fallback = MODEL_ROOT / "weights" / candidate.name
    if fallback.exists():
        logger.warning("checkpoint not found at {}, using {}", candidate, fallback)
        return str(fallback.resolve())

    return path


def _run(cmd: List[str], cwd: Path = REPO_ROOT, extra_env: dict[str, str] | None = None) -> None:
    logger.info("Executing: {}", " ".join(cmd))
    child_env = dict(os.environ)

    keys_to_remove = [
        key
        for key in child_env.keys()
        if key.startswith("DEBUGPY_") or key.startswith("PYDEVD_")
    ]
    for key in keys_to_remove:
        child_env.pop(key, None)

    pythonpath_value = child_env.get("PYTHONPATH")
    if pythonpath_value:
        sep = os.pathsep
        cleaned = [
            entry
            for entry in pythonpath_value.split(sep)
            if "ms-python.debugpy" not in entry and "debugpy" not in Path(entry).name
        ]
        if cleaned:
            child_env["PYTHONPATH"] = sep.join(cleaned)
        else:
            child_env.pop("PYTHONPATH", None)

    if extra_env:
        child_env.update(extra_env)

    subprocess.run(cmd, cwd=str(cwd), check=True, env=child_env)


def _run_train(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "engine" / "flows" / "train" / "mangr_train.py"),
        "--config",
        args.config,
        "--model-profile",
        args.model_profile,
    ]
    overrides = [*args.overrides, f"train.output_dir={run_root}"]
    cmd.extend(["--overrides", *overrides])
    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _run_eval(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval (can be provided via CLI)")
    run_root = Path(args.run_root)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "engine" / "flows" / "eval" / "mangr_eval.py"),
        "--config",
        args.config,
        "--model-profile",
        args.model_profile,
        "--checkpoint",
        checkpoint_path,
        "--device",
        args.device,
    ]
    overrides = [*args.overrides, f"train.output_dir={run_root}"]
    cmd.extend(["--overrides", *overrides])
    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _run_inference(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    inference_dir = run_root / "inference"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "engine" / "flows" / "inference" / "mangr_inference.py"),
        "--config",
        args.config,
        "--model-profile",
        args.model_profile,
        "--input-dir",
        args.input_dir,
        "--output-dir",
        str(inference_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]

    if getattr(args, "checkpoint", ""):
        cmd.extend(["--checkpoint", _resolve_checkpoint_path(args.checkpoint)])
    if getattr(args, "onnx_model", ""):
        cmd.extend(["--onnx-model", args.onnx_model])
    if args.overrides:
        cmd.extend(["--overrides", *args.overrides])

    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _run_export_onnx(args: argparse.Namespace) -> None:
    if not args.checkpoint or not args.onnx_model:
        raise ValueError("--checkpoint and --onnx-model are required for export-onnx")
    run_root = Path(args.run_root)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    cmd = [
        sys.executable,
        str(MODEL_ROOT / "tools" / "export_onnx.py"),
        "-c",
        str(MODEL_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml"),
        "-r",
        checkpoint_path,
        "-o",
        args.onnx_model,
        "--check",
        "--simplify",
    ]
    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _run_export_coco_rle(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    dataset_conf = args.dataset_conf or args.config
    if not dataset_conf:
        raise ValueError("--dataset-conf (or --config) is required for export-coco-rle")
    if not args.dataset_root:
        raise ValueError("--dataset_root is required for export-coco-rle")
    if not args.output_dir:
        raise ValueError("--output_dir is required for export-coco-rle")

    splits, default_ann_subdir, default_img_subdir = _load_dataset_export_settings(dataset_conf)

    selected_splits = args.splits if args.splits else splits
    ann_subdir = args.ann_subdir or default_ann_subdir
    img_subdir = args.img_subdir or default_img_subdir
    experiment_conf = args.experiment_conf or _resolve_experiment_conf_path(dataset_conf)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "infra" / "tools" / "export_coco_poly_2_rle.py"),
        "--conf_data",
        dataset_conf,
        "--dataset_root",
        args.dataset_root,
        "--output_dir",
        args.output_dir,
        "--ann_subdir",
        ann_subdir,
        "--img_subdir",
        img_subdir,
        "--splits",
        *selected_splits,
    ]
    if experiment_conf:
        cmd.extend(["--experiment_conf", experiment_conf])
    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _add_common_arguments(target_parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    target_parser.add_argument("--config", type=str, required=True)
    target_parser.add_argument("--model-profile", default=defaults["model_profile"], choices=["r18", "r50"])
    target_parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    target_parser.add_argument("--overrides", nargs="*", default=[])


def _register_train_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    train_parser = subparsers.add_parser("train", help="Run training")
    _add_common_arguments(train_parser, defaults)
    train_parser.set_defaults(handler=_run_train)


def _register_eval_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    _add_common_arguments(eval_parser, defaults)
    eval_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    eval_parser.add_argument("--device", type=str, default=defaults["device"])
    eval_parser.set_defaults(handler=_run_eval)


def _register_inference_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    infer_parser = subparsers.add_parser("inference", help="Run PyTorch inference")
    _add_common_arguments(infer_parser, defaults)
    infer_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    infer_parser.add_argument("--input-dir", type=str, default=defaults["input_dir"])
    infer_parser.add_argument("--device", type=str, default=defaults["device"])
    infer_parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    infer_parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    infer_parser.set_defaults(handler=_run_inference)


def _register_inference_onnx_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    infer_onnx_parser = subparsers.add_parser("inference-onnx", help="Run ONNX inference")
    _add_common_arguments(infer_onnx_parser, defaults)
    infer_onnx_parser.add_argument("--onnx-model", type=str, default=defaults["onnx_model"])
    infer_onnx_parser.add_argument("--input-dir", type=str, default=defaults["input_dir"])
    infer_onnx_parser.add_argument("--device", type=str, default=defaults["device"])
    infer_onnx_parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    infer_onnx_parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    infer_onnx_parser.set_defaults(handler=_run_inference)


def _register_export_onnx_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    export_onnx_parser = subparsers.add_parser("export-onnx", help="Export ONNX model")
    export_onnx_parser.add_argument("--config", type=str, required=True)
    export_onnx_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    export_onnx_parser.add_argument("--onnx-model", type=str, default=defaults["onnx_model"])
    export_onnx_parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    export_onnx_parser.set_defaults(handler=_run_export_onnx)


def _register_export_coco_rle_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    export_parser = subparsers.add_parser("export-coco-rle", help="Convert Supervisely rectangles to COCO RLE")
    export_parser.add_argument("--config", type=str, required=True)
    export_parser.add_argument("--dataset-conf", type=str, default=defaults["dataset_conf"])
    export_parser.add_argument("--experiment-conf", type=str, default=defaults["experiment_conf"])
    export_parser.add_argument("--dataset_root", type=str, default=defaults["dataset_root"])
    export_parser.add_argument("--output_dir", type=str, default=defaults["output_dir"])
    export_parser.add_argument("--splits", nargs="+", default=defaults["splits"])
    export_parser.add_argument("--ann_subdir", type=str, default=defaults["ann_subdir"])
    export_parser.add_argument("--img_subdir", type=str, default=defaults["img_subdir"])
    export_parser.set_defaults(handler=_run_export_coco_rle)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("action", choices=["train", "eval", "inference", "inference-onnx", "export-onnx", "export-coco-rle"])
    bootstrap.add_argument("--config", required=True)
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = _build_parser_defaults(bootstrap_args.config, bootstrap_args.action)

    parser = argparse.ArgumentParser(description="Run internal nn_framework flows")
    subparsers = parser.add_subparsers(dest="action", required=True)

    _register_train_parser(subparsers, defaults)
    _register_eval_parser(subparsers, defaults)
    _register_inference_parser(subparsers, defaults)
    _register_inference_onnx_parser(subparsers, defaults)
    _register_export_onnx_parser(subparsers, defaults)
    _register_export_coco_rle_parser(subparsers, defaults)

    args = parser.parse_args()
    args.config = str(_resolve_config_path(args.config))
    if getattr(args, "dataset_conf", None):
        args.dataset_conf = str(_resolve_config_path(args.dataset_conf))
    if getattr(args, "experiment_conf", None):
        args.experiment_conf = str(_resolve_config_path(args.experiment_conf))
    return args


def _prepare_run_layout(args: argparse.Namespace) -> Path:
    base_out = Path(args.output_dir).expanduser()
    if not base_out.is_absolute():
        base_out = (REPO_ROOT / base_out).resolve()

    run_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_root = base_out / run_name
    for subdir in ("logs", "configs", "inference", "dataset"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

    payload = {
        "action": args.action,
        "run_root": str(run_root),
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_") and k != "handler"},
    }
    with (run_root / "configs" / "execution.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    return run_root


def main() -> None:
    args = parse_args()
    run_root = _prepare_run_layout(args)
    args.run_root = str(run_root)
    os.environ["NN_FRAMEWORK_RUN_DIR"] = str(run_root)
    setup_logger(force=True)
    logger.info("Run directory: {}", run_root)
    args.handler(args)


if __name__ == "__main__":
    main()
