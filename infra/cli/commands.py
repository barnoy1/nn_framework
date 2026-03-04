from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from infra.utils.log import logger

from .config_defaults import load_dataset_export_settings, resolve_experiment_conf_path
from .constants import MODEL_ROOT, REPO_ROOT


def resolve_checkpoint_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    fallback = MODEL_ROOT / "weights" / candidate.name
    if fallback.exists():
        logger.warning("checkpoint not found at {}, using {}", candidate, fallback)
        return str(fallback.resolve())

    return path


def run_process(cmd: List[str], cwd: Path = REPO_ROOT, extra_env: dict[str, str] | None = None) -> None:
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


def run_train(args: argparse.Namespace) -> None:
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
    run_process(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def run_eval(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval (can be provided via CLI)")
    run_root = Path(args.run_root)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
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
    run_process(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def run_inference(args: argparse.Namespace) -> None:
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
        cmd.extend(["--checkpoint", resolve_checkpoint_path(args.checkpoint)])
    if getattr(args, "onnx_model", ""):
        cmd.extend(["--onnx-model", args.onnx_model])
    if args.overrides:
        cmd.extend(["--overrides", *args.overrides])

    run_process(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def run_export_onnx(args: argparse.Namespace) -> None:
    if not args.checkpoint or not args.onnx_model:
        raise ValueError("--checkpoint and --onnx-model are required for export-onnx")
    run_root = Path(args.run_root)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
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
    run_process(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def run_export_coco_rle(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    dataset_conf = args.dataset_conf or args.config
    if not dataset_conf:
        raise ValueError("--dataset-conf (or --config) is required for export-coco-rle")
    if not args.dataset_root:
        raise ValueError("--dataset_root is required for export-coco-rle")
    if not args.output_dir:
        raise ValueError("--output_dir is required for export-coco-rle")

    splits, default_ann_subdir, default_img_subdir = load_dataset_export_settings(dataset_conf)

    selected_splits = args.splits if args.splits else splits
    ann_subdir = args.ann_subdir or default_ann_subdir
    img_subdir = args.img_subdir or default_img_subdir
    experiment_conf = args.experiment_conf or resolve_experiment_conf_path(dataset_conf)

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
    run_process(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})
