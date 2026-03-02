from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    def on_train_start(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        pass

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        pass

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        pass


class WandBCallback(Callback):
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.wandb = None

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled:
            return
        import wandb

        self.wandb = wandb
        wandb.init(
            project=trainer.app_config.runtime.wandb_project,
            name=trainer.app_config.runtime.wandb_run_name,
            config=trainer.app_config.model.model_dump() | trainer.app_config.train.model_dump(),
        )

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        if self.wandb is not None and trainer.accelerator.is_main_process:
            self.wandb.log({"epoch": epoch, "step": step, **metrics}, step=trainer.global_step)

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if self.wandb is not None and trainer.accelerator.is_main_process:
            self.wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"epoch": epoch}, step=trainer.global_step)


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


class EMACallback(Callback):
    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        if trainer.ema_model is not None:
            trainer.ema_model.update(trainer.accelerator.unwrap_model(trainer.model))


class DynamicAugCallback(Callback):
    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        dataset = trainer.train_loader.dataset
        transforms = getattr(dataset, "transforms", None)
        if transforms is not None and hasattr(transforms, "update_augmentation"):
            transforms.update_augmentation(epoch=epoch, total_epochs=trainer.app_config.train.epochs)


class CallbackList:
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_train_start(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, epoch, step, metrics)

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, epoch, metrics)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
