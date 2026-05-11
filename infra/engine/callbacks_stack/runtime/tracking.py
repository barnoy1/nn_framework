from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import mlflow
import yaml

from ...artifacts.mlflow_tracking_helpers import (
    artifact_root,
    ensure_schema_up_to_date,
    extract_run_metadata_from_experiment_yaml,
    flatten_payload,
    resolve_run_folder_name,
    trim_param,
)
from infra.tracking import start_mlflow_ui_service
from ..core import Callback

if TYPE_CHECKING:
    from ...trainer import Trainer


class MLflowCallback(Callback):
    @staticmethod
    def _should_auto_open_browser() -> bool:
        if Path("/.dockerenv").exists():
            return False
        if str(os.environ.get("container", "")).strip().lower() in {
            "docker",
            "podman",
        }:
            return False
        return bool(str(os.environ.get("DISPLAY", "")).strip())

    @staticmethod
    def _resolve_model_name(trainer: "Trainer") -> str:
        source_root = str(getattr(trainer.app_config.model, "source_root", "") or "")
        source_root = source_root.strip().rstrip("/")
        if not source_root:
            return "model"
        return Path(source_root).name or source_root

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

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        tracking_dir = self.tracking_dir or (
            Path(trainer.app_config.train.output_dir) / "mlflow"
        )
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow_artifact_root = artifact_root(tracking_dir)
        experiment_name = self.experiment_name or trainer.experiment_name
        run_output_dir = Path(trainer.app_config.train.output_dir)
        run_folder_name = resolve_run_folder_name(
            run_output_dir=run_output_dir,
            tracking_dir=tracking_dir.resolve(),
        )
        model_name = self._resolve_model_name(trainer)
        run_base_name = self.run_name or experiment_name
        if model_name and model_name not in run_base_name:
            run_base_name = f"{run_base_name}__{model_name}"
        run_name = f"{run_base_name}__{run_folder_name}"
        backend = str(self.tracking_backend).strip().lower()
        tracking_uri: str
        if backend == "sqlite":
            sqlite_path = (tracking_dir / self.sqlite_db_name).resolve()
            tracking_uri = f"sqlite:///{sqlite_path}"
            ensure_schema_up_to_date(tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = tracking_dir.resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(
                experiment_name, artifact_location=mlflow_artifact_root.as_uri()
            )
            experiment = client.get_experiment(experiment_id)
        mlflow.set_experiment(experiment_name)

        if self.start_ui_service and trainer.accelerator.is_main_process:
            try:
                mlflow_url = start_mlflow_ui_service(
                    tracking_dir=tracking_dir,
                    host=self.ui_host,
                    port=self.ui_port,
                    logger_port=trainer.logger,
                    tracking_backend=self.tracking_backend,
                    sqlite_db_name=self.sqlite_db_name,
                )
                if experiment is not None:
                    experiment_url = (
                        f"{mlflow_url}/#/experiments/{experiment.experiment_id}"
                    )
                    if self._should_auto_open_browser():
                        webbrowser.open(experiment_url, new=0)
                    trainer.logger.info("MLflow experiment URL: {}", experiment_url)
            except Exception as error:
                trainer.logger.warning("MLflow UI service startup failed: {}", error)

        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.start_run(run_name=run_name)
            self._owns_run = True
        else:
            self._owns_run = False
            mlflow.set_tag("mlflow.runName", run_name)

        trainer.logger.info("MLflow tracking dir: {}", tracking_dir)
        trainer.logger.info("MLflow artifact root: {}", mlflow_artifact_root)
        execution_config = trainer.app_config.model_dump(mode="json")
        flattened = flatten_payload(execution_config)
        if flattened:
            items = list(flattened.items())
            chunk_size = 100
            for index in range(0, len(items), chunk_size):
                chunk = {
                    key: trim_param(value)
                    for key, value in items[index : index + chunk_size]
                }
                mlflow.log_params(chunk)
        experiment_config_path = getattr(trainer, "experiment_config_path", None)
        config_dir = Path(trainer.app_config.train.output_dir) / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        target = config_dir / "experiment.yaml"
        if experiment_config_path is not None:
            resolved_config_path = Path(experiment_config_path).resolve()
            if resolved_config_path.exists() and resolved_config_path.is_file():
                if target.resolve() != resolved_config_path:
                    target.write_text(
                        resolved_config_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
            else:
                config_yaml = yaml.safe_dump(
                    execution_config, sort_keys=True, allow_unicode=True
                )
                (config_dir / "experiment.yaml").write_text(
                    config_yaml, encoding="utf-8"
                )
        else:
            config_yaml = yaml.safe_dump(
                execution_config, sort_keys=True, allow_unicode=True
            )
            target.write_text(config_yaml, encoding="utf-8")

        metadata_tags = extract_run_metadata_from_experiment_yaml(target)
        if not metadata_tags:
            runtime_cfg = (
                execution_config.get("runtime")
                if isinstance(execution_config.get("runtime"), dict)
                else {}
            )
            fallback_description = str(runtime_cfg.get("description") or "").strip()
            fallback_source_root = str(
                execution_config.get("model", {}).get("source_root")
                if isinstance(execution_config.get("model"), dict)
                else ""
            ).strip()
            if fallback_description:
                metadata_tags["description"] = fallback_description
            if fallback_source_root:
                metadata_tags["model.source_root"] = fallback_source_root
                metadata_tags["model.name"] = Path(fallback_source_root).name

        if "description" in metadata_tags:
            mlflow.set_tag("description", metadata_tags["description"])
            mlflow.set_tag("runtime.description", metadata_tags["description"])
        mlflow.set_tag("model.name", metadata_tags.get("model.name", model_name))
        mlflow.set_tag(
            "model.source_root",
            metadata_tags.get(
                "model.source_root", str(trainer.app_config.model.source_root)
            ),
        )
        self._active = True

    def _step(self, raw_step: int) -> int:
        step = int(raw_step)
        if step <= self._last_step:
            step = self._last_step + 1
        self._last_step = step
        return step

    @staticmethod
    def _extract_epoch_loss_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for source_key, target_key in (
            ("train_loss", "train/total_loss"),
            ("val_loss", "val/total_loss"),
        ):
            value = metrics.get(source_key)
            if value is None:
                continue
            try:
                payload[target_key] = float(value)
            except (TypeError, ValueError):
                continue
        return payload

    def on_batch_end(
        self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]
    ) -> None:
        return

    def on_validation_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        return

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if self._active and trainer.accelerator.is_main_process:
            current_step = self._step(trainer.global_step)
            payload = {"epoch": float(epoch)}
            for key, value in metrics.items():
                try:
                    payload[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            mlflow.log_metrics(payload, step=current_step)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._active and self._owns_run:
            mlflow.end_run()
        self._active = False
        self._owns_run = False
