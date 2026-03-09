from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.common.logging.logger import logger

from infra.engine.flows.common.config_loader import get_execution_config
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
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()

def invoke(args) -> None:
    config_path = Path(str(args.config)).expanduser()
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()

    runtime = build_flow_runtime(overrides=args.overrides, config_path=args.config)
    app_config = get_execution_config(runtime=runtime)
    experiment_name = Path(args.config).stem
   
    if str(args.checkpoint).strip():
        state = runtime.wrapper.load_checkpoint_state(str(args.checkpoint))
        try:
            runtime.wrapper.validate_checkpoint_class_compatibility(runtime.built.model, state)
        except RuntimeError as exc:
            logger.warning(
                "Checkpoint compatibility warning during training warm-start: {}. "
                "Continuing with partial state load (classification heads may be re-initialized).",
                exc,
            )
        loaded, skipped, missing = runtime.wrapper.safe_load_state_dict(runtime.built.model, state)
        logger.info("Loaded training checkpoint tensors={}, skipped_shape={}, missing={}", loaded, skipped, missing)

    profile_train_and_val_dataset_distribution(runtime, logger)
    mlflow_cfg = app_config.runtime.visualization.mlflow
    run_output_dir = app_config.ensure_output_dir().resolve()
    shared_tracking_dir = run_output_dir.parent if "__" in run_output_dir.name else run_output_dir
    monitor_keys_cfg = app_config.train.metrics_key
    monitor_keys = [monitor_keys_cfg] if isinstance(monitor_keys_cfg, str) else list(monitor_keys_cfg)
    primary_monitor_key = monitor_keys[0]

    callbacks = CallbackList(
        [
            DynamicAugCallback(),
            EMACallback(),
            YoloStyleArtifactsCallback(enabled=True),
            CheckpointCallback(
                output_dir=app_config.ensure_output_dir(),
                save_every_n_epochs=app_config.train.save_every_n_epochs,
                monitor_key=primary_monitor_key,
                monitor_keys=monitor_keys,
            ),
            MLflowCallback(
                enabled=bool(mlflow_cfg.enabled),
                tracking_dir=shared_tracking_dir,
                experiment_name=app_config.runtime.mlflow_experiment_name or experiment_name,
                run_name=app_config.runtime.description or experiment_name,
                tracking_backend=str(mlflow_cfg.tracking_backend),
                sqlite_db_name=str(mlflow_cfg.sqlite_db_name),
                ui_host=str(mlflow_cfg.host),
                ui_port=int(mlflow_cfg.port),
                start_ui_service=bool(mlflow_cfg.start_service),
            ),
            ValidationVisualizationCallback(
                output_dir=app_config.ensure_output_dir(),
                num_samples=int(app_config.runtime.visualization.num_samples),
                experiment_name=experiment_name,
                tensorboard_enabled=bool(app_config.runtime.visualization.tensorboard.enabled),
                tensorboard_log_dir=str(app_config.runtime.visualization.tensorboard.log_dir),
                mlflow_enabled=bool(app_config.runtime.visualization.mlflow.enabled),
            ),
        ]
    )

    trainer = Trainer(
        app_config=app_config,
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
        experiment_config_path=config_path if config_path.exists() else None,
    )
    trainer.fit()

def main() -> None:
    args = parse_args()
    invoke(args)

if __name__ == "__main__":
    main()
