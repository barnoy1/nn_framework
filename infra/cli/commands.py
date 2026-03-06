from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from infra.common.logging import logger

from .config_defaults import load_dataset_export_settings, resolve_experiment_conf_path
from .constants import MODEL_ROOT


def resolve_checkpoint_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    fallback = MODEL_ROOT / "weights" / candidate.name
    if fallback.exists():
        logger.warning("checkpoint not found at {}, using {}", candidate, fallback)
        return str(fallback.resolve())

    return path


def run_train(args: argparse.Namespace) -> None:
    args.overrides = [*args.overrides, f"train.output_dir={Path(args.run_root)}"]

    from infra.engine.flows.train.mangr_train import invoke
    args = SimpleNamespace(**dict(
        config=args.config,
        checkpoint=args.checkpoint, 
        overrides=args.overrides
    ))
    invoke(args)


def run_eval(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval (can be provided via CLI)")
    run_root = Path(args.run_root)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    overrides = [*args.overrides, f"train.output_dir={run_root}"]

    from infra.engine.flows.eval.mangr_eval import invoke

    invoke(
        SimpleNamespace(
            config=args.config,
            checkpoint=checkpoint_path,
            device=args.device,
            vis_samples=getattr(args, "vis_samples", 16),
            score_thr=getattr(args, "score_thr", None),
            overrides=overrides,
        )
    )


def run_inference(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    inference_dir = run_root / "inference"
    checkpoint = ""
    if getattr(args, "checkpoint", ""):
        checkpoint = resolve_checkpoint_path(args.checkpoint)

    from infra.engine.flows.inference.mangr_inference import invoke

    invoke(
        SimpleNamespace(
            config=args.config,
            checkpoint=checkpoint,
            onnx_model=getattr(args, "onnx_model", ""),
            input_dir=args.input_dir,
            output_dir=str(inference_dir),
            device=args.device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            score_thr=float(getattr(args, "score_thr", 0.3)),
            overrides=args.overrides,
        )
    )


def run_export_onnx(args: argparse.Namespace) -> None:
    if not args.checkpoint or not args.onnx_model:
        raise ValueError("--checkpoint and --onnx-model are required for export-onnx")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    export_script = MODEL_ROOT / "tools" / "export_onnx.py"
    spec = importlib.util.spec_from_file_location("rtdetr_export_onnx", str(export_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load export script module: {export_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main(
        SimpleNamespace(
            config=str(MODEL_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml"),
            resume=checkpoint_path,
            output_file=args.onnx_model,
            input_size=640,
            check=True,
            simplify=True,
            update=None,
        )
    )


def run_export_coco_rle(args: argparse.Namespace) -> None:
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

    from infra.tools.export_coco_rle.app import invoke

    invoke(
        SimpleNamespace(
            conf_data=dataset_conf,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            ann_subdir=ann_subdir,
            img_subdir=img_subdir,
            splits=selected_splits,
            experiment_conf=experiment_conf,
            logging_level="info",
        )
    )
