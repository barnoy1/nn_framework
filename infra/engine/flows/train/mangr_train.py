from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.adapters import LoguruLoggerAdapter
from infra.engine.flows.common.runtime import build_flow_runtime
from infra.engine.flows.eval.dataset_profile import profile_train_and_val_dataset_distribution
from infra.engine.callbacks import (
    CallbackList,
    CheckpointCallback,
    DynamicAugCallback,
    EMACallback,
    WandBCallback,
)
from infra.engine.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework training manager")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_flow_runtime(model_profile=args.model_profile, overrides=args.overrides, config_path=args.config)
    experiment_name = Path(args.config).stem
    logger = LoguruLoggerAdapter()
    profile_train_and_val_dataset_distribution(runtime, logger)

    callbacks = CallbackList(
        [
            DynamicAugCallback(),
            EMACallback(),
            CheckpointCallback(
                output_dir=runtime.app_config.ensure_output_dir(),
                save_every_n_epochs=runtime.app_config.train.save_every_n_epochs,
            ),
            WandBCallback(enabled=bool(runtime.app_config.runtime.wandb_project)),
        ]
    )

    trainer = Trainer(
        app_config=runtime.app_config,
        model=runtime.built.model,
        criterion=runtime.built.criterion,
        postprocessor=runtime.built.postprocessor,
        optimizer=runtime.built.optimizer,
        scheduler=runtime.built.scheduler,
        train_loader=runtime.train_loader,
        val_loader=runtime.val_loader,
        callbacks=callbacks,
        ema_model=runtime.built.ema_model,
        model_wrapper=runtime.wrapper,
        experiment_name=experiment_name,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
