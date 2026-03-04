from __future__ import annotations

import argparse
from typing import Any, Callable

from .commands import run_eval, run_export_coco_rle, run_export_onnx, run_inference, run_train
from .config_defaults import build_parser_defaults, resolve_config_path

ActionHandler = Callable[[argparse.Namespace], None]


def add_common_arguments(target_parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    target_parser.add_argument("--config", type=str, required=True)
    target_parser.add_argument("--model-profile", default=defaults["model_profile"], choices=["r18", "r50"])
    target_parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    target_parser.add_argument("--overrides", nargs="*", default=[])


def register_train_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    train_parser = subparsers.add_parser("train", help="Run training")
    add_common_arguments(train_parser, defaults)
    train_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    train_parser.set_defaults(handler=run_train)


def register_eval_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    add_common_arguments(eval_parser, defaults)
    eval_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    eval_parser.add_argument("--device", type=str, default=defaults["device"])
    eval_parser.set_defaults(handler=run_eval)


def register_inference_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    infer_parser = subparsers.add_parser("inference", help="Run PyTorch inference")
    add_common_arguments(infer_parser, defaults)
    infer_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    infer_parser.add_argument("--input-dir", type=str, default=defaults["input_dir"])
    infer_parser.add_argument("--device", type=str, default=defaults["device"])
    infer_parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    infer_parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    infer_parser.set_defaults(handler=run_inference)


def register_inference_onnx_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    infer_onnx_parser = subparsers.add_parser("inference-onnx", help="Run ONNX inference")
    add_common_arguments(infer_onnx_parser, defaults)
    infer_onnx_parser.add_argument("--onnx-model", type=str, default=defaults["onnx_model"])
    infer_onnx_parser.add_argument("--input-dir", type=str, default=defaults["input_dir"])
    infer_onnx_parser.add_argument("--device", type=str, default=defaults["device"])
    infer_onnx_parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    infer_onnx_parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    infer_onnx_parser.set_defaults(handler=run_inference)


def register_export_onnx_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    export_onnx_parser = subparsers.add_parser("export-onnx", help="Export ONNX model")
    export_onnx_parser.add_argument("--config", type=str, required=True)
    export_onnx_parser.add_argument("--checkpoint", type=str, default=defaults["checkpoint"])
    export_onnx_parser.add_argument("--onnx-model", type=str, default=defaults["onnx_model"])
    export_onnx_parser.add_argument("--output-dir", type=str, default=defaults["output_dir"])
    export_onnx_parser.set_defaults(handler=run_export_onnx)


def register_export_coco_rle_parser(subparsers: argparse._SubParsersAction, defaults: dict[str, Any]) -> None:
    export_parser = subparsers.add_parser("export-coco-rle", help="Convert Supervisely rectangles to COCO RLE")
    export_parser.add_argument("--config", type=str, required=True)
    export_parser.add_argument("--dataset-conf", type=str, default=defaults["dataset_conf"])
    export_parser.add_argument("--experiment-conf", type=str, default=defaults["experiment_conf"])
    export_parser.add_argument("--dataset_root", type=str, default=defaults["dataset_root"])
    export_parser.add_argument("--output_dir", type=str, default=defaults["output_dir"])
    export_parser.add_argument("--splits", nargs="+", default=defaults["splits"])
    export_parser.add_argument("--ann_subdir", type=str, default=defaults["ann_subdir"])
    export_parser.add_argument("--img_subdir", type=str, default=defaults["img_subdir"])
    export_parser.set_defaults(handler=run_export_coco_rle)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("action", choices=["train", "eval", "inference", "inference-onnx", "export-onnx", "export-coco-rle"])
    bootstrap.add_argument("--config", required=True)
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = build_parser_defaults(bootstrap_args.config, bootstrap_args.action)

    parser = argparse.ArgumentParser(description="Run internal nn_framework flows")
    subparsers = parser.add_subparsers(dest="action", required=True)

    register_train_parser(subparsers, defaults)
    register_eval_parser(subparsers, defaults)
    register_inference_parser(subparsers, defaults)
    register_inference_onnx_parser(subparsers, defaults)
    register_export_onnx_parser(subparsers, defaults)
    register_export_coco_rle_parser(subparsers, defaults)

    args = parser.parse_args()
    args.config = str(resolve_config_path(args.config))
    if getattr(args, "dataset_conf", None):
        args.dataset_conf = str(resolve_config_path(args.dataset_conf))
    if getattr(args, "experiment_conf", None):
        args.experiment_conf = str(resolve_config_path(args.experiment_conf))
    return args
