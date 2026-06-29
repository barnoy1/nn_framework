from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from infra.common import RuntimePathResolver

from .config_defaults import (
    load_config_payload,
    load_dataset_export_settings,
    resolve_config_path,
    resolve_experiment_conf_path,
)
from .constants import REPO_ROOT


REPO_ROOT_TOKEN = "@REPO_ROOT/"
MODEL_ROOT_TOKEN = "@MODEL_ROOT/"


def _expand_runtime_env_tokens(path: str) -> str:
    return RuntimePathResolver.expand_runtime_tokens(path)


def _model_section(payload: dict) -> dict:
    adapter_cfg = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    model_cfg = adapter_cfg.get("model") if isinstance(adapter_cfg.get("model"), dict) else {}
    return model_cfg


def resolve_model_root(config_path: str) -> Path:
    payload = load_config_payload(config_path)
    model_cfg = _model_section(payload)
    source_root = str(model_cfg.get("source_root") or "").strip()
    if not source_root:
        raise ValueError("adapter.model.source_root must be set in the experiment config")

    candidate = Path(source_root).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Model source root not found: {candidate}")
    return candidate


def resolve_model_config_path(config_path: str) -> Path:
    payload = load_config_payload(config_path)
    model_cfg = _model_section(payload)
    model_config_path = str(model_cfg.get("model_config_path") or "").strip()
    if not model_config_path:
        raise ValueError("adapter.model.model_config_path must be set in the experiment config")

    model_root = resolve_model_root(config_path)
    candidate = Path(model_config_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    resolved = (model_root / candidate).resolve()
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Model config not found from experiment config: model_config_path={model_config_path}, resolved={resolved}"
    )


def resolve_runtime_path(path: str, *, config_path: str | None = None) -> str:
    raw = _expand_runtime_env_tokens(path)
    if not raw:
        return raw

    if raw.startswith(REPO_ROOT_TOKEN):
        return str((REPO_ROOT / raw[len(REPO_ROOT_TOKEN) :]).resolve())

    if raw.startswith(MODEL_ROOT_TOKEN):
        if not config_path:
            return raw
        model_root = resolve_model_root(config_path)
        return str((model_root / raw[len(MODEL_ROOT_TOKEN) :]).resolve())

    return raw


def resolve_checkpoint_path(path: str, *, config_path: str | None = None) -> str:
    resolved_input = resolve_runtime_path(path, config_path=config_path)
    extra_roots = []
    if config_path:
        model_root = resolve_model_root(config_path)
        extra_roots.append(model_root / "weights")

    resolver = RuntimePathResolver(repo_root=REPO_ROOT, extra_search_roots=extra_roots)
    try:
        return str(resolver.resolve_checkpoint(resolved_input))
    except FileNotFoundError:
        return str(Path(resolved_input).expanduser())


def run_train(args: argparse.Namespace) -> None:
    args.overrides = [*args.overrides, f"engine.execution.common.output_dir={Path(args.run_root)}"]

    from infra.engine.flows.train.mangr_train import invoke

    args = SimpleNamespace(
        **dict(config=args.config, checkpoint=args.checkpoint, overrides=args.overrides)
    )
    invoke(args)


def run_eval(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval (can be provided via CLI)")
    run_root = Path(args.run_root)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, config_path=args.config)
    overrides = [*args.overrides, f"engine.execution.common.output_dir={run_root}"]

    from infra.engine.flows.eval.mangr_eval import invoke

    invoke(
        SimpleNamespace(
            config=args.config,
            checkpoint=checkpoint_path,
            device=args.device,
            vis_samples=getattr(args, "vis_samples", 16),
            score_thr=getattr(args, "score_thr", None),
            gt_data=getattr(args, "gt_data", []),
            allow_partial=getattr(args, "allow_partial", False),
            overrides=overrides,
        )
    )


def run_inference(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    inference_dir = run_root / "inference"
    checkpoint = ""
    if getattr(args, "checkpoint", ""):
        checkpoint = resolve_checkpoint_path(args.checkpoint, config_path=args.config)
    onnx_model = resolve_runtime_path(
        getattr(args, "onnx_model", ""), config_path=args.config
    )

    from infra.engine.flows.inference.mangr_inference import invoke

    invoke(
        SimpleNamespace(
            config=args.config,
            checkpoint=checkpoint,
            onnx_model=onnx_model,
            input_dir=args.input_dir,
            output_dir=str(inference_dir),
            device=args.device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            score_thr=float(getattr(args, "score_thr", 0.3)),
            allow_partial=getattr(args, "allow_partial", False),
            overrides=args.overrides,
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

    splits, default_ann_subdir, default_img_subdir = load_dataset_export_settings(
        dataset_conf
    )

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
