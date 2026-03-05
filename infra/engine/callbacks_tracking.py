from __future__ import annotations

import json
import webbrowser
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
    def _artifact_root(tracking_dir: Path) -> Path:
        root = (tracking_dir / "mlruns").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

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

    @staticmethod
    def _normalize_registered_model_name(value: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/"
        cleaned = "".join(ch if ch in allowed else "_" for ch in value)
        cleaned = cleaned.strip("._-/")
        return cleaned or "model"

    @staticmethod
    def _resolve_run_folder_name(run_output_dir: Path, tracking_dir: Path) -> str:
        resolved_output_dir = run_output_dir.resolve()
        output_parts = resolved_output_dir.name.split("__", 1)
        if len(output_parts) == 2 and output_parts[1].strip():
            return output_parts[1].strip()

        parent_parts = resolved_output_dir.parent.name.split("__", 1)
        if len(parent_parts) == 2 and parent_parts[1].strip():
            return parent_parts[1].strip()

        if tracking_dir.name == "mlflow" and tracking_dir.parent.name == "visualization":
            return tracking_dir.parent.parent.name
        return tracking_dir.parent.name

    def _register_current_model(
        self,
        *,
        trainer: "Trainer",
        client: "mlflow.tracking.MlflowClient",
        experiment_name: str,
        execution_config: Dict,
    ) -> None:
        if not trainer.accelerator.is_main_process:
            return

        model_cfg = execution_config.get("model") if isinstance(execution_config, dict) else None
        if not isinstance(model_cfg, dict):
            return

        source_root = str(model_cfg.get("source_root") or "").strip()
        model_config_path = str(model_cfg.get("model_config_path") or "").strip()
        if not source_root and not model_config_path:
            return

        model_info = {
            "source_root": source_root,
            "model_config_path": model_config_path,
        }
        mlflow.log_text(
            yaml.safe_dump(model_info, sort_keys=True, allow_unicode=True),
            artifact_file="model/definition.yaml",
        )
        mlflow.set_tags({
            "model.source_root": source_root,
            "model.model_config_path": model_config_path,
        })

        active_run = mlflow.active_run()
        if active_run is None:
            return

        model_cfg_name = Path(model_config_path).stem if model_config_path else "model"
        registered_model_name = self._normalize_registered_model_name(
            f"{experiment_name}__{model_cfg_name}"
        )
        model_uri = f"runs:/{active_run.info.run_id}/model"

        class _IdentityModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                return model_input

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_IdentityModel(),
            metadata=model_info,
            input_example={
                "source_root": source_root,
                "model_config_path": model_config_path,
            },
        )

        try:
            client.create_registered_model(registered_model_name)
        except Exception:
            pass

        try:
            client.create_model_version(
                name=registered_model_name,
                source=model_uri,
                run_id=active_run.info.run_id,
            )
            trainer.logger.info(
                "Registered MLflow model={} from run={}",
                registered_model_name,
                active_run.info.run_id,
            )
        except Exception as error:
            trainer.logger.warning("MLflow model registration skipped: {}", error)

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        tracking_dir = self.tracking_dir or (Path(trainer.app_config.train.output_dir) / "mlflow")
        tracking_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = self._artifact_root(tracking_dir)
        experiment_name = self.experiment_name or trainer.experiment_name
        run_output_dir = Path(trainer.app_config.train.output_dir)
        run_folder_name = self._resolve_run_folder_name(
            run_output_dir=run_output_dir,
            tracking_dir=tracking_dir.resolve(),
        )
        run_base_name = self.run_name or experiment_name
        run_name = f"{run_base_name}__{run_folder_name}"
        backend = str(self.tracking_backend).strip().lower()
        tracking_uri: str
        if backend == "sqlite":
            sqlite_path = (tracking_dir / self.sqlite_db_name).resolve()
            tracking_uri = f"sqlite:///{sqlite_path}"
            mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = tracking_dir.resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
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
                    experiment_url = f"{mlflow_url}/#/experiments/{experiment.experiment_id}"
                    webbrowser.open_new_tab(experiment_url)
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
        trainer.logger.info("MLflow artifact root: {}", artifact_root)
        execution_config = trainer.app_config.model_dump(mode="json")
        self._register_current_model(
            trainer=trainer,
            client=client,
            experiment_name=experiment_name,
            execution_config=execution_config,
        )
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
