from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ...artifacts import (
    build_epoch_row,
    compute_detection_scores,
    render_training_artifact_plots,
    save_confusion_matrix_plots,
    save_labels_plot,
    write_results_csv,
)
from ...flows.eval.dataset_profile import collect_class_frequency
from ..core import Callback

if TYPE_CHECKING:
    from ...trainer import Trainer


class YoloStyleArtifactsCallback(Callback):
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._results_csv: Optional[Path] = None
        self._rows: List[Dict[str, float]] = []
        self._last_precision = 0.0
        self._last_recall = 0.0
        self._last_f1 = 0.0
        self._last_accuracy = 0.0

    def on_train_start(self, trainer: "Trainer") -> None:
        if not self.enabled or not trainer.accelerator.is_main_process:
            return
        output_root = Path(trainer.app_config.train.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        self._results_csv = output_root / "results.csv"
        self._save_labels_plot(trainer)

    def _save_labels_plot(self, trainer: "Trainer") -> None:
        output_root = Path(trainer.app_config.train.output_dir)
        train_counts, train_names = collect_class_frequency(
            trainer.train_loader.dataset
        )
        val_counts, val_names = collect_class_frequency(trainer.val_loader.dataset)
        save_labels_plot(
            output_root=output_root,
            train_counts=train_counts,
            train_names=train_names,
            val_counts=val_counts,
            val_names=val_names,
        )
        legacy_jpg = output_root / "labels.jpg"
        if legacy_jpg.exists():
            legacy_jpg.unlink(missing_ok=True)

    def on_validation_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if not self.enabled or not trainer.accelerator.is_main_process:
            return
        matrix = getattr(trainer, "last_validation_confusion_matrix", None)
        names = getattr(trainer, "last_validation_confusion_labels", None)
        if matrix is None or names is None:
            return

        output_root = Path(trainer.app_config.train.output_dir)
        save_confusion_matrix_plots(
            output_root=output_root, matrix=matrix, labels=names
        )
        scores = compute_detection_scores(matrix)
        self._last_precision = float(scores["precision"])
        self._last_recall = float(scores["recall"])
        self._last_f1 = float(scores["f1"])
        self._last_accuracy = float(scores["accuracy"])

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if not self.enabled or not trainer.accelerator.is_main_process:
            return
        optimizer_lrs = [
            float(group.get("lr", 0.0))
            for group in getattr(trainer.optimizer, "param_groups", [])
        ]
        row = build_epoch_row(
            epoch=epoch,
            metrics=metrics,
            optimizer_lrs=optimizer_lrs,
            precision=self._last_precision,
            recall=self._last_recall,
            f1=self._last_f1,
            accuracy=self._last_accuracy,
        )
        self._rows.append(row)
        self._write_results_csv()
        render_training_artifact_plots(
            output_root=Path(trainer.app_config.train.output_dir), rows=self._rows
        )

    def _write_results_csv(self) -> None:
        if self._results_csv is None:
            return
        write_results_csv(self._results_csv, self._rows)
