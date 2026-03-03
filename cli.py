from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List

import yaml

from utils.log import logger, setup_logger

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "raw_models" / "RT-DETR" / "rtdetrv2_pytorch"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ActionHandler = Callable[[argparse.Namespace], None]


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
        str(REPO_ROOT / "flows" / "train" / "mangr_train.py"),
        "--model-profile",
        args.model_profile,
    ]
    overrides = [*args.overrides, f"train.output_dir={run_root}"]
    cmd.extend(["--overrides", *overrides])
    _run(cmd, extra_env={"NN_FRAMEWORK_RUN_DIR": str(run_root)})


def _run_eval(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "flows" / "eval" / "mangr_eval.py"),
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
        str(REPO_ROOT / "flows" / "inference" / "mangr_inference.py"),
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


def _add_common_arguments(target_parser: argparse.ArgumentParser) -> None:
    target_parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    target_parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "out"))
    target_parser.add_argument("--overrides", nargs="*", default=[])


def _register_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Run training")
    _add_common_arguments(train_parser)
    train_parser.set_defaults(handler=_run_train)


def _register_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    _add_common_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--device", type=str, default="cuda")
    eval_parser.set_defaults(handler=_run_eval)


def _register_inference_parser(subparsers: argparse._SubParsersAction) -> None:
    infer_parser = subparsers.add_parser("inference", help="Run PyTorch inference")
    _add_common_arguments(infer_parser)
    infer_parser.add_argument("--checkpoint", type=str, required=True)
    infer_parser.add_argument("--input-dir", type=str, required=True)
    infer_parser.add_argument("--device", type=str, default="cuda")
    infer_parser.add_argument("--batch-size", type=int, default=1)
    infer_parser.add_argument("--num-workers", type=int, default=2)
    infer_parser.set_defaults(handler=_run_inference)


def _register_inference_onnx_parser(subparsers: argparse._SubParsersAction) -> None:
    infer_onnx_parser = subparsers.add_parser("inference-onnx", help="Run ONNX inference")
    _add_common_arguments(infer_onnx_parser)
    infer_onnx_parser.add_argument("--onnx-model", type=str, required=True)
    infer_onnx_parser.add_argument("--input-dir", type=str, required=True)
    infer_onnx_parser.add_argument("--device", type=str, default="cuda")
    infer_onnx_parser.add_argument("--batch-size", type=int, default=1)
    infer_onnx_parser.add_argument("--num-workers", type=int, default=2)
    infer_onnx_parser.set_defaults(handler=_run_inference)


def _register_export_onnx_parser(subparsers: argparse._SubParsersAction) -> None:
    export_onnx_parser = subparsers.add_parser("export-onnx", help="Export ONNX model")
    export_onnx_parser.add_argument("--checkpoint", type=str, required=True)
    export_onnx_parser.add_argument("--onnx-model", type=str, required=True)
    export_onnx_parser.set_defaults(handler=_run_export_onnx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal nn_framework flows")
    subparsers = parser.add_subparsers(dest="action", required=True)

    _register_train_parser(subparsers)
    _register_eval_parser(subparsers)
    _register_inference_parser(subparsers)
    _register_inference_onnx_parser(subparsers)
    _register_export_onnx_parser(subparsers)

    return parser.parse_args()


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
