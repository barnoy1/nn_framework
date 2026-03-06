from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..trainer import Trainer


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

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


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

    def on_train_end(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
