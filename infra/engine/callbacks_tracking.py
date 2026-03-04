from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import mlflow

from .callbacks_base import Callback

if TYPE_CHECKING:
    from .trainer import Trainer


class MLflowCallback(Callback):
    def __init__(self, enabled: bool = False, tracking_dir: Optional[Path] = None, experiment_name: Optional[str] = None, run_name: Optional[str] = None) -> None:
        self.enabled = enabled
        self.tracking_dir = tracking_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._active = False
        self._last_step = -1

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        tracking_dir = self.tracking_dir or (Path(trainer.app_config.train.output_dir) / "mlflow")
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        experiment_name = self.experiment_name or trainer.experiment_name
        run_name = self.run_name or trainer.experiment_name
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(trainer.app_config.model.model_dump() | trainer.app_config.train.model_dump())
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

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._active:
            mlflow.end_run()
            self._active = False
