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
        monitor_keys: list[str] | None = None,
        monitor_mode: str = "auto",
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_mode = str(monitor_mode).strip().lower()
        keys = monitor_keys if monitor_keys is not None else [monitor_key]
        normalized_keys = [str(key).strip() for key in keys if str(key).strip()]
        self.monitor_keys = normalized_keys if normalized_keys else ["map"]
        self.best_values: Dict[str, float] = {}

    @staticmethod
    def _metric_slug(metric_key: str) -> str:
        slug = "".join(
            char if (char.isalnum() or char in "-_") else "_"
            for char in str(metric_key).strip()
        )
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_") or "metric"

    def _resolve_monitor_mode(self, monitor_key: str) -> str:
        if self.monitor_mode in {"min", "max"}:
            return self.monitor_mode
        lowered_key = str(monitor_key).lower()
        if "loss" in lowered_key:
            return "min"
        return "max"

    def _is_better(self, monitor_key: str, value: float) -> bool:
        mode = self._resolve_monitor_mode(monitor_key)
        previous = self.best_values.get(monitor_key)
        if previous is None:
            return True
        if mode == "min":
            return value < float(previous)
        return value > float(previous)

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

    def _load_best_epoch_entries(self) -> list[dict]:
        best_epoch_path = self.output_dir / "best_epoch.json"
        if not best_epoch_path.exists():
            return []
        try:
            payload = json.loads(best_epoch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        return []

    def _write_best_epoch_entries(self, entries: list[dict]) -> None:
        (self.output_dir / "best_epoch.json").write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _sync_best_eval_pointer(
        self, epoch: int, monitor_key: str, monitor_value: float
    ) -> None:
        eval_root = self.output_dir / "inference" / "eval"
        if not eval_root.exists():
            return
        epoch_suffix = f"__epoch_{epoch + 1:04d}.png"
        epoch_files = [
            candidate
            for candidate in eval_root.rglob(f"*{epoch_suffix}")
            if candidate.is_file()
        ]
        if not epoch_files:
            return

        metric_slug = self._metric_slug(monitor_key)
        best_checkpoint = self.checkpoint_dir / f"best_{metric_slug}.pt"
        pointer_payload = {
            "best_epoch": int(epoch + 1),
            "monitor_key": str(monitor_key),
            "monitor_value": float(monitor_value),
            "source_eval_dir": str(eval_root),
            "best_checkpoint": str(best_checkpoint),
        }
        entries = [
            entry
            for entry in self._load_best_epoch_entries()
            if str(entry.get("monitor_key")) != str(monitor_key)
        ]
        entries.append(pointer_payload)
        self._write_best_epoch_entries(entries)

        best_dir = self.output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        metric_best_dir = best_dir / metric_slug
        metric_best_dir.mkdir(parents=True, exist_ok=True)
        for candidate in epoch_files:
            target = metric_best_dir / candidate.name
            shutil.copy2(candidate, target)

        (metric_best_dir / "pointer.json").write_text(
            json.dumps(pointer_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if not trainer.accelerator.is_main_process:
            return

        if (epoch + 1) % self.save_every_n_epochs == 0:
            self._save(trainer, f"checkpoint_epoch_{epoch + 1}.pt")
        self._save(trainer, "last.pt")

        for monitor_key in self.monitor_keys:
            raw_value = metrics.get(monitor_key)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if self._is_better(monitor_key, value):
                self.best_values[monitor_key] = value
                metric_slug = self._metric_slug(monitor_key)
                self._save(trainer, f"best_{metric_slug}.pt")
                if monitor_key == self.monitor_keys[0]:
                    self._save(trainer, "best.pt")
                self._sync_best_eval_pointer(
                    epoch=epoch, monitor_key=monitor_key, monitor_value=value
                )
