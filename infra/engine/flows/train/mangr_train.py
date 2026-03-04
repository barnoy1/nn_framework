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
    MLflowCallback,
    ValidationVisualizationCallback,
    YoloStyleArtifactsCallback,
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
    mlflow_cfg = runtime.app_config.runtime.visualization.mlflow

    callbacks = CallbackList(
        [
            DynamicAugCallback(),
            EMACallback(),
            YoloStyleArtifactsCallback(enabled=True),
            CheckpointCallback(
                output_dir=runtime.app_config.ensure_output_dir(),
                save_every_n_epochs=runtime.app_config.train.save_every_n_epochs,
            ),
            MLflowCallback(
                enabled=bool(mlflow_cfg.enabled),
                tracking_dir=runtime.app_config.ensure_output_dir() / str(mlflow_cfg.mlflow_dir),
                experiment_name=runtime.app_config.runtime.mlflow_experiment_name or experiment_name,
                run_name=runtime.app_config.runtime.mlflow_run_name or experiment_name,
                tracking_backend=str(mlflow_cfg.tracking_backend),
                sqlite_db_name=str(mlflow_cfg.sqlite_db_name),
                ui_host=str(mlflow_cfg.host),
                ui_port=int(mlflow_cfg.port),
                start_ui_service=bool(mlflow_cfg.start_service),
            ),
            ValidationVisualizationCallback(
                output_dir=runtime.app_config.ensure_output_dir(),
                num_samples=int(runtime.app_config.runtime.visualization.num_samples),
                experiment_name=experiment_name,
                tensorboard_enabled=bool(runtime.app_config.runtime.visualization.tensorboard.enabled),
                tensorboard_log_dir=str(runtime.app_config.runtime.visualization.tensorboard.log_dir),
                mlflow_enabled=bool(runtime.app_config.runtime.visualization.mlflow.enabled),
            ),
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
