from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch

from .callbacks_base import Callback

if TYPE_CHECKING:
    from .trainer import Trainer


class CheckpointCallback(Callback):
    def __init__(self, output_dir: Path, save_every_n_epochs: int = 1, monitor_key: str = "map") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_key = monitor_key
        self.best_value = float("-inf")

    def _save(self, trainer: "Trainer", name: str) -> None:
        state = {
            "model": trainer.accelerator.unwrap_model(trainer.model).state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "config": trainer.app_config.model_dump(),
        }
        if trainer.ema_model is not None:
            state["ema"] = trainer.ema_model.state_dict()
        torch.save(state, self.output_dir / name)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process:
            return

        if (epoch + 1) % self.save_every_n_epochs == 0:
            self._save(trainer, f"checkpoint_epoch_{epoch + 1}.pt")
        self._save(trainer, "last.pt")

        value = float(metrics.get(self.monitor_key, float("-inf")))
        if value > self.best_value:
            self.best_value = value
            self._save(trainer, "best.pt")
