from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import numpy as np
from PIL import Image

from infra.tracking import create_visualization_logger
from ..core import Callback
from .visualization_helpers import (
    compose_grid,
    log_accumulated_eval_history_artifacts,
    log_batch_artifacts,
    log_dataset_artifacts,
    log_evaluation_artifacts,
    log_output_artifacts,
    sync_execution_tree_artifacts,
)

if TYPE_CHECKING:
    from ...trainer import Trainer


class ValidationVisualizationCallback(Callback):
    def __init__(
        self,
        output_dir: Path,
        num_samples: int,
        experiment_name: str,
        tensorboard_enabled: bool,
        tensorboard_log_dir: str,
        mlflow_enabled: bool,
    ) -> None:
        self.output_dir = output_dir
        self.num_samples = max(0, int(num_samples))
        self.experiment_name = experiment_name
        self.tensorboard_enabled = tensorboard_enabled
        self.tensorboard_log_dir = tensorboard_log_dir
        self.mlflow_enabled = mlflow_enabled
        self._logger = None
        self._save_dir = self.output_dir / "inference" / "validation"
        self._last_logged_artifacts: set[Path] = set()
        self._artifact_mtimes: dict[Path, int] = {}

    @staticmethod
    def _extract_epoch_loss_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for source_key, target_key in (("train_loss", "train/total_loss"), ("val_loss", "val/total_loss")):
            value = metrics.get(source_key)
            if value is None:
                continue
            try:
                payload[target_key] = float(value)
            except (TypeError, ValueError):
                continue
        return payload

    def on_train_start(self, trainer: "Trainer") -> None:
        if not trainer.accelerator.is_main_process:
            return
        vis_cfg = trainer.app_config.runtime.visualization
        output_root_resolved = self.output_dir.resolve()
        shared_tracking_dir = output_root_resolved.parent if "__" in output_root_resolved.name else output_root_resolved
        self._logger = create_visualization_logger(
            output_root=self.output_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=self.tensorboard_enabled,
            tensorboard_log_dir=self.tensorboard_log_dir,
            tensorboard_host=str(vis_cfg.tensorboard.host),
            tensorboard_port=int(vis_cfg.tensorboard.port),
            tensorboard_start_service=bool(vis_cfg.tensorboard.start_service),
            mlflow_enabled=self.mlflow_enabled,
            mlflow_dir=str(shared_tracking_dir),
            mlflow_tracking_backend=str(vis_cfg.mlflow.tracking_backend),
            mlflow_sqlite_db_name=str(vis_cfg.mlflow.sqlite_db_name),
            mlflow_host=str(vis_cfg.mlflow.host),
            mlflow_port=int(vis_cfg.mlflow.port),
            mlflow_start_service=bool(vis_cfg.mlflow.start_service),
            execution_config=trainer.app_config.model_dump(mode="json"),
            logger_port=trainer.logger,
        )

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        return

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process or self._logger is None:
            return

        matrix = getattr(trainer, "last_validation_confusion_matrix", None)
        labels = list(getattr(trainer, "last_validation_confusion_labels", []) or [])
        if matrix is not None and labels:
            self._logger.log_text(
                tag="val/confusion_labels",
                text=json.dumps(labels, ensure_ascii=False),
                step=epoch + 1,
            )

        metrics_path = (self.output_dir / "metrics.json").resolve()
        if metrics_path.exists():
            self._logger.log_text(
                tag="eval/metrics.json",
                text=f"path={metrics_path}",
                step=epoch + 1,
            )

        log_evaluation_artifacts(output_dir=self.output_dir, epoch=epoch, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        log_accumulated_eval_history_artifacts(output_dir=self.output_dir, logger=self._logger)
        log_dataset_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        log_batch_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        log_output_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        sync_execution_tree_artifacts(output_dir=self.output_dir, logger=self._logger, artifact_mtimes=self._artifact_mtimes)

        samples = list(getattr(trainer, "last_validation_visual_samples", []) or [])[: self.num_samples]
        if not samples:
            return

        panel = compose_grid(samples)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._save_dir / f"val_epoch_{epoch + 1:04d}.jpg"
        Image.fromarray(panel).save(output_path)

        self._logger.log_image(tag="train/val_visualization_grid", image=panel, step=epoch + 1)
        trainer.logger.info("Saved validation visualization grid ({} samples) to {}", len(samples), output_path)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process or self._logger is None:
            return
        numeric_metrics = self._extract_epoch_loss_metrics(metrics)
        if numeric_metrics:
            self._logger.log_metrics(metrics=numeric_metrics, step=epoch + 1)
        log_dataset_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        log_batch_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        log_output_artifacts(output_dir=self.output_dir, logger=self._logger, logged_artifacts=self._last_logged_artifacts)
        sync_execution_tree_artifacts(output_dir=self.output_dir, logger=self._logger, artifact_mtimes=self._artifact_mtimes)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._logger is not None:
            sync_execution_tree_artifacts(
                output_dir=self.output_dir,
                logger=self._logger,
                artifact_mtimes=self._artifact_mtimes,
            )
            self._logger.close()
            self._logger = None
