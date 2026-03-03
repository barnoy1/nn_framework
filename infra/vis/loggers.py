from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np


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


class WandbVisualizationLogger:
    def __init__(self, project: str, run_name: str, run_dir: Path, entity: Optional[str] = None):
        import wandb

        self._wandb = wandb
        target_dir = run_dir.resolve()
        init_dir = target_dir.parent if target_dir.name == "wandb" else target_dir
        init_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(init_dir))
        self._run = wandb.init(
            project=project,
            name=run_name,
            dir=str(init_dir),
            entity=entity,
            reinit="finish_previous",
        )
        self._last_step = -1

    def _monotonic_step(self, step: int) -> int:
        resolved = int(step)
        if resolved <= self._last_step:
            resolved = self._last_step + 1
        self._last_step = resolved
        return resolved

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        resolved_step = self._monotonic_step(step)
        self._wandb.log({tag: self._wandb.Image(image)}, step=resolved_step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        resolved_step = self._monotonic_step(step)
        self._wandb.log({key: float(value) for key, value in metrics.items()}, step=resolved_step)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


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
    wandb_enabled: bool,
    wandb_dir: str,
    wandb_entity: Optional[str],
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

    if wandb_enabled:
        try:
            resolved_wandb_dir = (output_root / wandb_dir).resolve()
            loggers.append(
                WandbVisualizationLogger(
                    project=experiment_name,
                    run_name=experiment_name,
                    run_dir=resolved_wandb_dir,
                    entity=wandb_entity,
                )
            )
            logger_port.info(
                "W&B visualization logging enabled entity={} project={} run={} dir={}",
                wandb_entity,
                experiment_name,
                experiment_name,
                resolved_wandb_dir,
            )
        except Exception as error:
            logger_port.warning("W&B visualization logger unavailable: {}", error)

    if not loggers:
        return NullVisualizationLogger()

    if len(loggers) == 1:
        return loggers[0]

    return CompositeVisualizationLogger(loggers=loggers)
