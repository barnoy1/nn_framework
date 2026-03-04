from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
from PIL import Image


class VisualizationLogger(Protocol):
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None: ...

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class NullVisualizationLogger:
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        return

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        return

    def close(self) -> None:
        return


class TensorBoardVisualizationLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        chw = np.transpose(image, (2, 0, 1)) if image.ndim == 3 else image
        self._writer.add_image(tag=tag, img_tensor=chw, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(tag=key, scalar_value=float(value), global_step=step)

    def close(self) -> None:
        self._writer.close()


class MlflowVisualizationLogger:
    def __init__(self, experiment_name: str, run_name: str, tracking_dir: Path):
        import mlflow

        self._mlflow = mlflow
        self._tracking_dir = tracking_dir.resolve()
        self._tracking_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow.set_tracking_uri(self._tracking_dir.as_uri())
        self._mlflow.set_experiment(experiment_name)
        self._run = self._mlflow.start_run(run_name=run_name)
        self._last_step = -1

    def _monotonic_step(self, step: int) -> int:
        resolved = int(step)
        if resolved <= self._last_step:
            resolved = self._last_step + 1
        self._last_step = resolved
        return resolved

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        resolved_step = self._monotonic_step(step)
        image_dir = self._tracking_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_name = tag.replace("/", "_")
        image_path = image_dir / f"{image_name}_{resolved_step:06d}.png"
        Image.fromarray(image).save(image_path)
        self._mlflow.log_artifact(str(image_path), artifact_path="images")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        resolved_step = self._monotonic_step(step)
        for key, value in metrics.items():
            self._mlflow.log_metric(key, float(value), step=resolved_step)

    def close(self) -> None:
        if self._run is not None:
            self._mlflow.end_run()


@dataclass
class CompositeVisualizationLogger:
    loggers: List[VisualizationLogger]

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        for logger in self.loggers:
            logger.log_image(tag=tag, image=image, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics=metrics, step=step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def create_visualization_logger(
    *,
    output_root: Path,
    experiment_name: str,
    tensorboard_enabled: bool,
    tensorboard_log_dir: str,
    mlflow_enabled: bool,
    mlflow_dir: str,
    logger_port,
) -> VisualizationLogger:
    loggers: List[VisualizationLogger] = []

    if tensorboard_enabled:
        try:
            resolved = (output_root / tensorboard_log_dir).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            loggers.append(TensorBoardVisualizationLogger(log_dir=resolved))
            logger_port.info("TensorBoard visualization logging enabled at {}", resolved)
        except Exception as error:
            logger_port.warning("TensorBoard visualization logger unavailable: {}", error)

    if mlflow_enabled:
        try:
            resolved_mlflow_dir = (output_root / mlflow_dir).resolve()
            loggers.append(
                MlflowVisualizationLogger(
                    experiment_name=experiment_name,
                    run_name=experiment_name,
                    tracking_dir=resolved_mlflow_dir,
                )
            )
            logger_port.info(
                "MLflow visualization logging enabled experiment={} run={} dir={}",
                experiment_name,
                experiment_name,
                resolved_mlflow_dir,
            )
        except Exception as error:
            logger_port.warning("MLflow visualization logger unavailable: {}", error)

    if not loggers:
        return NullVisualizationLogger()

    if len(loggers) == 1:
        return loggers[0]

    return CompositeVisualizationLogger(loggers=loggers)
