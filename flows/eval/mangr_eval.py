from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nn_framework.flows.common.runtime import build_flow_runtime
from nn_framework.engine.callbacks import CallbackList
from nn_framework.engine.trainer import Trainer
from nn_framework.utils.log import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework evaluation manager")
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-class-mismatch", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides)

    state = runtime.wrapper.load_checkpoint_state(args.checkpoint)
    runtime.wrapper.validate_checkpoint_class_compatibility(
        runtime.built.model,
        state,
        allow_mismatch=args.allow_class_mismatch,
    )
    loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
    logger.info("Loaded checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    trainer = Trainer(
        app_config=runtime.app_config,
        model=runtime.built.model,
        criterion=runtime.built.criterion,
        postprocessor=runtime.built.postprocessor,
        optimizer=runtime.built.optimizer,
        scheduler=runtime.built.scheduler,
        train_loader=runtime.train_loader,
        val_loader=runtime.val_loader,
        callbacks=CallbackList([]),
        ema_model=runtime.built.ema_model,
        model_wrapper=runtime.wrapper,
    )

    metrics = trainer.validate(epoch=0)
    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("{}: {}", key, value)


if __name__ == "__main__":
    main()
