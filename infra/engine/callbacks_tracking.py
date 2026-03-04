from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import mlflow
import yaml

from .callbacks_base import Callback
from ..vis import start_mlflow_ui_service

if TYPE_CHECKING:
    from .trainer import Trainer


class MLflowCallback(Callback):
    def __init__(
        self,
        enabled: bool = False,
        tracking_dir: Optional[Path] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_backend: str = "sqlite",
        sqlite_db_name: str = "mlflow.db",
        ui_host: str = "127.0.0.1",
        ui_port: int = 5000,
        start_ui_service: bool = True,
    ) -> None:
        self.enabled = enabled
        self.tracking_dir = tracking_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_backend = str(tracking_backend)
        self.sqlite_db_name = str(sqlite_db_name)
        self.ui_host = str(ui_host)
        self.ui_port = int(ui_port)
        self.start_ui_service = bool(start_ui_service)
        self._active = False
        self._owns_run = False
        self._last_step = -1

    @staticmethod
    def _flatten_payload(payload: Dict, prefix: str = "") -> Dict[str, str]:
        flattened: Dict[str, str] = {}
        for key, value in payload.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(MLflowCallback._flatten_payload(value, prefix=current))
            elif isinstance(value, (list, tuple)):
                flattened[current] = json.dumps(value, ensure_ascii=False)
            else:
                flattened[current] = str(value)
        return flattened

    @staticmethod
    def _trim_param(value: str, max_len: int = 490) -> str:
        text = str(value)
        if len(text) <= max_len:
            return text
        return f"{text[:max_len]}..."

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        tracking_dir = self.tracking_dir or (Path(trainer.app_config.train.output_dir) / "mlflow")
        tracking_dir.mkdir(parents=True, exist_ok=True)
        backend = str(self.tracking_backend).strip().lower()
        if backend == "sqlite":
            sqlite_path = (tracking_dir / self.sqlite_db_name).resolve()
            tracking_uri = f"sqlite:///{sqlite_path}"
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        if self.start_ui_service and trainer.accelerator.is_main_process:
            try:
                start_mlflow_ui_service(
                    tracking_dir=tracking_dir,
                    host=self.ui_host,
                    port=self.ui_port,
                    logger_port=trainer.logger,
                    tracking_backend=self.tracking_backend,
                    sqlite_db_name=self.sqlite_db_name,
                )
            except Exception as error:
                trainer.logger.warning("MLflow UI service startup failed: {}", error)
        experiment_name = self.experiment_name or trainer.experiment_name
        run_name = self.run_name or trainer.experiment_name
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.start_run(run_name=run_name)
            self._owns_run = True
        else:
            self._owns_run = False
        execution_config = trainer.app_config.model_dump(mode="json")
        flattened = self._flatten_payload(execution_config)
        if flattened:
            items = list(flattened.items())
            chunk_size = 100
            for index in range(0, len(items), chunk_size):
                chunk = {
                    key: self._trim_param(value)
                    for key, value in items[index : index + chunk_size]
                }
                mlflow.log_params(chunk)
        config_yaml = yaml.safe_dump(execution_config, sort_keys=True, allow_unicode=True)
        mlflow.log_text(config_yaml, artifact_file="config/config.yaml")
        self._active = True

    def _step(self, raw_step: int) -> int:
        step = int(raw_step)
        if step <= self._last_step:
            step = self._last_step + 1
        self._last_step = step
        return step

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        if self._active and trainer.accelerator.is_main_process:
            current_step = self._step(trainer.global_step)
            mlflow.log_metric("epoch", float(epoch), step=current_step)
            mlflow.log_metric("step", float(step), step=current_step)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value), step=current_step)

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if self._active and trainer.accelerator.is_main_process:
            current_step = self._step(trainer.global_step)
            mlflow.log_metric("epoch", float(epoch), step=current_step)
            for key, value in metrics.items():
                mlflow.log_metric(f"val/{key}", float(value), step=current_step)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if self._active and trainer.accelerator.is_main_process:
            current_step = self._step(trainer.global_step)
            mlflow.log_metric("epoch", float(epoch), step=current_step)
            for key, value in metrics.items():
                mlflow.log_metric(str(key), float(value), step=current_step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._active and self._owns_run:
            mlflow.end_run()
        self._active = False
        self._owns_run = False
