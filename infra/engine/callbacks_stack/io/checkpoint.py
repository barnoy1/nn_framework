from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch

from ..core import Callback

if TYPE_CHECKING:
    from ...trainer import Trainer


class CheckpointCallback(Callback):
    def __init__(
        self,
        output_dir: Path,
        save_every_n_epochs: int = 1,
        monitor_key: str = "map",
        monitor_mode: str = "auto",
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_key = monitor_key
        self.monitor_mode = str(monitor_mode).strip().lower()
        self.best_value = None

    def _resolve_monitor_mode(self) -> str:
        if self.monitor_mode in {"min", "max"}:
            return self.monitor_mode
        lowered_key = self.monitor_key.lower()
        if "loss" in lowered_key:
            return "min"
        return "max"

    def _is_better(self, value: float) -> bool:
        mode = self._resolve_monitor_mode()
        if self.best_value is None:
            return True
        if mode == "min":
            return value < float(self.best_value)
        return value > float(self.best_value)

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
        torch.save(state, self.checkpoint_dir / name)

    def _sync_best_eval_pointer(self, epoch: int, monitor_value: float) -> None:
        eval_root = self.output_dir / "inference" / "eval"
        if not eval_root.exists():
            return
        epoch_suffix = f"__epoch_{epoch + 1:04d}.png"
        epoch_files = [candidate for candidate in eval_root.rglob(f"*{epoch_suffix}") if candidate.is_file()]
        if not epoch_files:
            return

        pointer_payload = {
            "best_epoch": int(epoch + 1),
            "monitor_key": str(self.monitor_key),
            "monitor_value": float(monitor_value),
            "source_eval_dir": str(eval_root),
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
        }
        (self.output_dir / "best_epoch.json").write_text(
            json.dumps(pointer_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        best_dir = self.output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        for candidate in epoch_files:
            target = best_dir / candidate.name
            shutil.copy2(candidate, target)

        (best_dir / "pointer.json").write_text(
            json.dumps(pointer_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process:
            return

        if (epoch + 1) % self.save_every_n_epochs == 0:
            self._save(trainer, f"checkpoint_epoch_{epoch + 1}.pt")
        self._save(trainer, "last.pt")

        raw_value = metrics.get(self.monitor_key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        if self._is_better(value):
            self.best_value = value
            self._save(trainer, "best.pt")
            self._sync_best_eval_pointer(epoch=epoch, monitor_value=value)
