from __future__ import annotations

from typing import Dict, TYPE_CHECKING

from ..core import Callback

if TYPE_CHECKING:
    from ...trainer import Trainer


class EMACallback(Callback):
    def on_batch_end(
        self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]
    ) -> None:
        if trainer.ema_model is not None:
            trainer.ema_model.update(trainer.accelerator.unwrap_model(trainer.model))


class DynamicAugCallback(Callback):
    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        root_dataset = trainer.train_loader.dataset
        datasets = list(getattr(root_dataset, "datasets", []) or [root_dataset])
        for dataset in datasets:
            transforms = getattr(dataset, "transforms", None)
            if transforms is not None and hasattr(transforms, "update_augmentation"):
                transforms.update_augmentation(
                    epoch=epoch, total_epochs=trainer.total_epochs
                )
