from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from typing import Dict

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.common.logging.logger import logger
from infra.engine.flows.common.config_loader import get_execution_config
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.engine.flows.eval.dataset_profile import (
    model_num_classes,
    profile_dataset_distribution,
)
from infra.engine.flows.eval.eval_artifacts import run_eval_artifacts
from infra.engine.training import save_eval_batch_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework evaluation manager")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vis-samples", type=int, default=16)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def invoke(args: argparse.Namespace) -> None:
    runtime = build_flow_runtime(overrides=args.overrides, config_path=args.config)
    app_config = get_execution_config(runtime=runtime)
    app_config.train.use_ema = False
    app_config.train.mixed_precision = "no"
    runtime.built.ema_model = None

    profile_dataset_distribution(runtime, logger)

    output_root = app_config.ensure_output_dir()
    eval_batch_num_samples = int(app_config.runtime.visualization.num_samples)
    saved_eval_batches = 0
    for step, (images, targets) in enumerate(runtime.val_loader):
        if step >= 3:
            break
        save_eval_batch_visualization(
            output_root=output_root,
            images=images,
            targets=targets,
            step=step,
            epoch_suffix=None,
            num_samples=eval_batch_num_samples,
        )
        saved_eval_batches += 1
    if saved_eval_batches > 0:
        logger.info(
            "Saved {} eval batch visualizations to {}", saved_eval_batches, output_root
        )

    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(
        runtime.built.model, state
    )
    logger.info(
        "Loaded checkpoint tensors={}, skipped_shape={}, missing={}",
        loaded,
        skipped,
        missing,
    )

    net_classes = model_num_classes(runtime.built.model)
    if net_classes is not None:
        configured_mapping = app_config.data.mapping or {}
        configured_label_ids = sorted(
            {int(label_id) for label_id in configured_mapping.values()}
        )
        if configured_label_ids:
            out_of_range = [
                label_id
                for label_id in configured_label_ids
                if label_id < 0 or label_id >= net_classes
            ]
            if out_of_range:
                logger.warning(
                    "validation label ids {} are out of model class range [0, {}]. Evaluation AP will be invalid.",
                    out_of_range,
                    net_classes - 1,
                )
            else:
                logger.info(
                    "Validation uses mapped label ids {} within model class range [0, {}].",
                    configured_label_ids,
                    net_classes - 1,
                )

    experiment_name = Path(args.config).stem
    device = torch.device(args.device)
    class_id_to_name = runtime.built.class_id_to_name
    diagnostics: Dict[str, object] = {}
    score_thr = (
        float(args.score_thr)
        if args.score_thr is not None
        else float(app_config.runtime.common.score_threshold)
    )
    metrics = run_eval_artifacts(
        app_config=app_config,
        model=runtime.built.model,
        postprocessor=runtime.built.postprocessor,
        device=device,
        logger=logger,
        class_id_to_name=class_id_to_name,
        experiment_name=experiment_name,
        vis_samples=args.vis_samples,
        score_thr=score_thr,
        image_epoch_suffix=None,
        write_metrics_json=True,
        diagnostics=diagnostics,
    )

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("{}: {}", key, value)


def main() -> None:
    args = parse_args()
    invoke(args)


if __name__ == "__main__":
    main()
