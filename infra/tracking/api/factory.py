from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .interfaces import (
    CompositeExperimentTracker,
    NullExperimentTracker,
    ExperimentTracker,
)
from .mlflow_backend import MlflowExperimentTracker
from .tb_backend import TensorBoardExperimentTracker
from ..service_launchers import start_mlflow_ui_service, start_tensorboard_service


def _resolve_mlflow_run_base_name(
    experiment_name: str, execution_config: Dict[str, Any] | None
) -> str:
    model_name = ""
    engine_cfg = execution_config.get("engine") if isinstance(execution_config, dict) else None
    execution_cfg = engine_cfg.get("execution") if isinstance(engine_cfg, dict) else None
    description = str((execution_cfg or {}).get("description") or "").strip()
    base = description or str(experiment_name)

    adapter_cfg = execution_config.get("adapter") if isinstance(execution_config, dict) else None
    model_cfg = adapter_cfg.get("model") if isinstance(adapter_cfg, dict) else None
    source_root = str((model_cfg or {}).get("source_root") or "").strip().rstrip("/")
    if source_root:
        model_name = Path(source_root).name or source_root

    if model_name and model_name not in base:
        return f"{base}__{model_name}"
    return base


def create_experiment_tracker(
    *,
    output_root: Path,
    experiment_name: str,
    tensorboard_enabled: bool,
    tensorboard_log_dir: str,
    tensorboard_host: str = "127.0.0.1",
    tensorboard_port: int = 6006,
    tensorboard_start_service: bool = True,
    mlflow_enabled: bool,
    mlflow_dir: str,
    mlflow_tracking_backend: str = "sqlite",
    mlflow_sqlite_db_name: str = "mlflow.db",
    mlflow_host: str = "127.0.0.1",
    mlflow_port: int = 5000,
    mlflow_start_service: bool = True,
    execution_config: Dict[str, Any] | None = None,
    logger_port=None,
) -> ExperimentTracker:
    loggers: List[ExperimentTracker] = []

    if tensorboard_enabled:
        try:
            resolved = (output_root / tensorboard_log_dir).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            loggers.append(TensorBoardExperimentTracker(log_dir=resolved))
            if logger_port is not None:
                logger_port.info(
                    "TensorBoard visualization logging enabled at {}", resolved
                )
            if tensorboard_start_service and logger_port is not None:
                tensorboard_url = start_tensorboard_service(
                    log_dir=resolved,
                    host=str(tensorboard_host),
                    port=int(tensorboard_port),
                    logger_port=logger_port,
                )
                logger_port.info(
                    "TensorBoard dashboard URL: {}/#scalars", tensorboard_url
                )
        except Exception as error:
            if logger_port is not None:
                logger_port.warning(
                    "TensorBoard visualization logger unavailable: {}", error
                )

    if mlflow_enabled:
        try:
            resolved_mlflow_dir = (output_root / mlflow_dir).resolve()
            run_base_name = _resolve_mlflow_run_base_name(
                experiment_name=experiment_name,
                execution_config=execution_config,
            )
            mlflow_logger = MlflowExperimentTracker(
                experiment_name=experiment_name,
                run_name=run_base_name,
                tracking_dir=resolved_mlflow_dir,
                run_context_dir=output_root,
                tracking_backend=str(mlflow_tracking_backend),
                sqlite_db_name=str(mlflow_sqlite_db_name),
                execution_config=execution_config,
            )
            loggers.append(mlflow_logger)
            if logger_port is not None:
                logger_port.info(
                    "MLflow visualization logging enabled experiment={} run={} dir={}",
                    experiment_name,
                    experiment_name,
                    resolved_mlflow_dir,
                )
                if mlflow_logger.experiment_id and mlflow_logger.run_id:
                    logger_port.info(
                        "MLflow run path: /#/experiments/{}/runs/{}",
                        mlflow_logger.experiment_id,
                        mlflow_logger.run_id,
                    )
            if mlflow_start_service and logger_port is not None:
                mlflow_url = start_mlflow_ui_service(
                    tracking_dir=resolved_mlflow_dir,
                    host=str(mlflow_host),
                    port=int(mlflow_port),
                    logger_port=logger_port,
                    tracking_backend=str(mlflow_tracking_backend),
                    sqlite_db_name=str(mlflow_sqlite_db_name),
                )
                logger_port.info("MLflow dashboard URL: {}", mlflow_url)
                if mlflow_logger.experiment_id and mlflow_logger.run_id:
                    logger_port.info(
                        "MLflow run URL: {}/#/experiments/{}/runs/{}",
                        mlflow_url,
                        mlflow_logger.experiment_id,
                        mlflow_logger.run_id,
                    )
        except Exception as error:
            if logger_port is not None:
                logger_port.warning(
                    "MLflow visualization logger unavailable: {}", error
                )

    if not loggers:
        return NullExperimentTracker()
    tracker = loggers[0] if len(loggers) == 1 else CompositeExperimentTracker(
        loggers=loggers
    )
    if execution_config:
        tracker.log_execution_config(execution_config)
    return tracker
