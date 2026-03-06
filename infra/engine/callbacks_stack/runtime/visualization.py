from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import numpy as np
from PIL import Image

from infra.tracking import create_visualization_logger
from ..core import Callback
from .visualization_helpers import (
    build_validation_metric_payload,
    compose_grid,
    compute_detection_scores,
    log_dataset_artifacts,
    log_evaluation_artifacts,
    log_output_artifacts,
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
        if not trainer.accelerator.is_main_process or self._logger is None:
            return
        self._logger.log_metrics(metrics=metrics, step=trainer.global_step)

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process:
            return
        if self._logger is None:
            return

        validation_payload = build_validation_metric_payload(metrics)
        if validation_payload:
            self._logger.log_metrics(metrics=validation_payload, step=epoch + 1)

        matrix = getattr(trainer, "last_validation_confusion_matrix", None)
        labels = list(getattr(trainer, "last_validation_confusion_labels", []) or [])
        if matrix is not None and labels:
            scores = compute_detection_scores(matrix)
            self._logger.log_metrics(
                metrics={f"val/confusion/{key}": float(value) for key, value in scores.items()},
                step=epoch + 1,
            )
            self._logger.log_metrics(
                metrics={f"evaluation/{key}": float(value) for key, value in scores.items()},
                step=epoch + 1,
            )
            self._logger.log_text(
                tag="val/confusion_labels",
                text=json.dumps(labels, ensure_ascii=False),
                step=epoch + 1,
            )

        eval_dir = self.output_dir / "inference" / "eval"
        for artifact_name in ("metrics.json", "detections.json"):
            artifact_path = eval_dir / artifact_name
            resolved_artifact = artifact_path.resolve()
            if resolved_artifact.exists():
                self._logger.log_text(
                    tag=f"eval/{artifact_name}",
                    text=f"path={resolved_artifact}",
                    step=epoch + 1,
                )

        log_evaluation_artifacts(output_dir=self.output_dir, epoch=epoch, logger=self._logger)
        log_dataset_artifacts(
            output_dir=self.output_dir,
            logger=self._logger,
            logged_artifacts=self._last_logged_artifacts,
        )
        log_output_artifacts(
            output_dir=self.output_dir,
            logger=self._logger,
            logged_artifacts=self._last_logged_artifacts,
        )

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
        numeric_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if numeric_metrics:
            self._logger.log_metrics(metrics=numeric_metrics, step=epoch + 1)
        log_dataset_artifacts(
            output_dir=self.output_dir,
            logger=self._logger,
            logged_artifacts=self._last_logged_artifacts,
        )
        log_output_artifacts(
            output_dir=self.output_dir,
            logger=self._logger,
            logged_artifacts=self._last_logged_artifacts,
        )

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None
