from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from typing import Dict

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.adapters import LoguruLoggerAdapter
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.engine.flows.eval.dataset_profile import model_num_classes, profile_dataset_distribution
from infra.engine.flows.eval.shared import run_eval_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework evaluation manager")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vis-samples", type=int, default=16)
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = LoguruLoggerAdapter()
    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides, config_path=args.config)
    runtime.app_config.train.use_ema = False
    runtime.app_config.train.mixed_precision = "no"
    runtime.built.ema_model = None

    profile_dataset_distribution(runtime, logger)

    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    net_classes = model_num_classes(runtime.built.model)
    if net_classes is not None:
        configured_mapping = runtime.app_config.data.mapping or {}
        configured_label_ids = sorted({int(label_id) for label_id in configured_mapping.values()})
        if configured_label_ids:
            out_of_range = [label_id for label_id in configured_label_ids if label_id < 0 or label_id >= net_classes]
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
    metrics = run_eval_artifacts(
        app_config=runtime.app_config,
        model=runtime.built.model,
        postprocessor=runtime.built.postprocessor,
        device=device,
        logger=logger,
        class_id_to_name=class_id_to_name,
        experiment_name=experiment_name,
        vis_samples=args.vis_samples,
        score_thr=args.score_thr,
        image_epoch_suffix=None,
        write_metrics_json=True,
    )

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("{}: {}", key, value)


if __name__ == "__main__":
    main()
