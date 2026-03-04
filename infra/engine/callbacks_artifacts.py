from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .callbacks_base import Callback
from .flows.eval.dataset_profile import collect_class_frequency

if TYPE_CHECKING:
    from .trainer import Trainer


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
        train_counts, train_names = collect_class_frequency(trainer.train_loader.dataset)
        val_counts, val_names = collect_class_frequency(trainer.val_loader.dataset)
        class_ids = sorted(set(train_counts.keys()) | set(val_counts.keys()))
        if not class_ids:
            return

        train_values = [int(train_counts.get(class_id, 0)) for class_id in class_ids]
        val_values = [int(val_counts.get(class_id, 0)) for class_id in class_ids]
        labels = [train_names.get(class_id, val_names.get(class_id, str(class_id))) for class_id in class_ids]

        x = np.arange(len(class_ids), dtype=np.float32)
        width = 0.4
        plt.figure(figsize=(max(10, len(class_ids) * 0.35), 7))
        plt.bar(x - width / 2, train_values, width=width, label="train")
        plt.bar(x + width / 2, val_values, width=width, label="val")
        plt.xticks(x, [f"{cid}:{name}" for cid, name in zip(class_ids, labels)], rotation=45, ha="right")
        plt.ylabel("instances")
        plt.title("labels")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / "labels.png", dpi=180)
        dataset_dir = output_root / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(dataset_dir / "labels.png", dpi=180)
        plt.close()
        legacy_jpg = output_root / "labels.jpg"
        if legacy_jpg.exists():
            legacy_jpg.unlink(missing_ok=True)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    def _compute_detection_scores(self, matrix: np.ndarray) -> Dict[str, float]:
        if matrix is None or matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        class_count = int(matrix.shape[0] - 1)
        core = matrix[:class_count, :class_count]
        tp = float(np.trace(core))
        pred_non_bg = float(matrix[:, :class_count].sum())
        gt_non_bg = float(matrix[:class_count, :].sum())
        fp = max(0.0, pred_non_bg - tp)
        fn = max(0.0, gt_non_bg - tp)
        precision = self._safe_ratio(tp, tp + fp)
        recall = self._safe_ratio(tp, tp + fn)
        f1 = self._safe_ratio(2.0 * precision * recall, precision + recall)
        accuracy = self._safe_ratio(tp, tp + fp + fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def _save_confusion_matrix_plot(self, output_root: Path, matrix: np.ndarray, labels: List[str]) -> None:
        fig_size = max(6, min(18, 0.25 * len(labels) + 6))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.title("confusion_matrix")
        plt.colorbar(fraction=0.046, pad=0.04)
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)
        plt.ylabel("true")
        plt.xlabel("pred")
        plt.tight_layout()
        plt.savefig(output_root / "confusion_matrix.png", dpi=180)
        plt.close()

    def _save_confusion_matrix_normalized_plot(self, output_root: Path, matrix: np.ndarray, labels: List[str]) -> None:
        row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
        normalized = np.divide(
            matrix.astype(np.float64),
            np.where(row_sums > 0.0, row_sums, 1.0),
            out=np.zeros_like(matrix, dtype=np.float64),
        )
        fig_size = max(6, min(18, 0.25 * len(labels) + 6))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        plt.title("confusion_matrix_normalized")
        plt.colorbar(fraction=0.046, pad=0.04)
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)
        plt.ylabel("true")
        plt.xlabel("pred")
        plt.tight_layout()
        plt.savefig(output_root / "confusion_matrix_normalized.png", dpi=180)
        plt.close()

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not self.enabled or not trainer.accelerator.is_main_process:
            return
        matrix = getattr(trainer, "last_validation_confusion_matrix", None)
        names = getattr(trainer, "last_validation_confusion_labels", None)
        if matrix is None or names is None:
            return

        output_root = Path(trainer.app_config.train.output_dir)
        self._save_confusion_matrix_plot(output_root=output_root, matrix=matrix, labels=names)
        self._save_confusion_matrix_normalized_plot(output_root=output_root, matrix=matrix, labels=names)
        scores = self._compute_detection_scores(matrix)
        self._last_precision = float(scores["precision"])
        self._last_recall = float(scores["recall"])
        self._last_f1 = float(scores["f1"])
        self._last_accuracy = float(scores["accuracy"])

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not self.enabled or not trainer.accelerator.is_main_process:
            return
        optimizer_lrs = [float(group.get("lr", 0.0)) for group in getattr(trainer.optimizer, "param_groups", [])]
        lr0 = optimizer_lrs[0] if len(optimizer_lrs) > 0 else 0.0
        lr1 = optimizer_lrs[1] if len(optimizer_lrs) > 1 else lr0
        lr2 = optimizer_lrs[2] if len(optimizer_lrs) > 2 else lr1
        row = {
            "epoch": float(epoch + 1),
            "time": float(epoch + 1),
            "train/box_loss": float(metrics.get("train_loss", 0.0)),
            "train/cls_loss": float(metrics.get("train_cls_loss", 0.0)),
            "train/dfl_loss": float(metrics.get("train_dfl_loss", 0.0)),
            "train/custom_loss": float(metrics.get("train_custom_loss", 0.0)),
            "metrics/precision(B)": float(self._last_precision),
            "metrics/recall(B)": float(self._last_recall),
            "metrics/F1(B)": float(self._last_f1),
            "metrics/accuracy(B)": float(self._last_accuracy),
            "metrics/mAP50(B)": float(metrics.get("val_bbox_map_50", 0.0)),
            "metrics/mAP50-95(B)": float(metrics.get("val_bbox_map", metrics.get("val_map", 0.0))),
            "val/box_loss": float(metrics.get("val_box_loss", 0.0)),
            "val/cls_loss": float(metrics.get("val_cls_loss", 0.0)),
            "val/dfl_loss": float(metrics.get("val_dfl_loss", 0.0)),
            "val/custom_loss": float(metrics.get("val_custom_loss", 0.0)),
            "lr/pg0": float(lr0),
            "lr/pg1": float(lr1),
            "lr/pg2": float(lr2),
            "val/bbox_map_75": float(metrics.get("val_bbox_map_75", 0.0)),
            "val/bbox_mar_100": float(metrics.get("val_bbox_mar_100", 0.0)),
        }
        self._rows.append(row)
        self._write_results_csv()
        self._plot_results(trainer)
        self._plot_bbox_metrics(trainer)
        self._plot_box_curves(trainer)

    def _write_results_csv(self) -> None:
        if self._results_csv is None:
            return
        fieldnames = [
            "epoch",
            "time",
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "train/custom_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "val/custom_loss",
            "lr/pg0",
            "lr/pg1",
            "lr/pg2",
            "metrics/F1(B)",
            "metrics/accuracy(B)",
            "val/bbox_map_75",
            "val/bbox_mar_100",
        ]
        with self._results_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def _plot_results(self, trainer: "Trainer") -> None:
        if not self._rows:
            return
        output_root = Path(trainer.app_config.train.output_dir)
        epochs = [row["epoch"] for row in self._rows]
        train_box_loss = [row["train/box_loss"] for row in self._rows]
        train_cls_loss = [row["train/cls_loss"] for row in self._rows]
        train_dfl_loss = [row["train/dfl_loss"] for row in self._rows]
        precision = [row["metrics/precision(B)"] for row in self._rows]
        recall = [row["metrics/recall(B)"] for row in self._rows]
        val_box_loss = [row["val/box_loss"] for row in self._rows]
        val_cls_loss = [row["val/cls_loss"] for row in self._rows]
        val_dfl_loss = [row["val/dfl_loss"] for row in self._rows]
        map50 = [row["metrics/mAP50(B)"] for row in self._rows]
        map5095 = [row["metrics/mAP50-95(B)"] for row in self._rows]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
        axes = axes.flatten()
        axes[0].plot(epochs, train_box_loss, marker="o")
        axes[0].set_title("train/box_loss")
        axes[0].set_xlabel("epoch")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, train_cls_loss, marker="o")
        axes[1].set_title("train/cls_loss")
        axes[1].set_xlabel("epoch")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, train_dfl_loss, marker="o")
        axes[2].set_title("train/dfl_loss")
        axes[2].set_xlabel("epoch")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(epochs, precision, marker="o")
        axes[3].set_ylim(0.0, 1.0)
        axes[3].set_title("metrics/precision(B)")
        axes[3].set_xlabel("epoch")
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(epochs, recall, marker="o")
        axes[4].set_ylim(0.0, 1.0)
        axes[4].set_title("metrics/recall(B)")
        axes[4].set_xlabel("epoch")
        axes[4].grid(True, alpha=0.3)

        axes[5].plot(epochs, val_box_loss, marker="o")
        axes[5].set_title("val/box_loss")
        axes[5].set_xlabel("epoch")
        axes[5].grid(True, alpha=0.3)

        axes[6].plot(epochs, val_cls_loss, marker="o")
        axes[6].set_title("val/cls_loss")
        axes[6].set_xlabel("epoch")
        axes[6].grid(True, alpha=0.3)

        axes[7].plot(epochs, val_dfl_loss, marker="o")
        axes[7].set_title("val/dfl_loss")
        axes[7].set_xlabel("epoch")
        axes[7].grid(True, alpha=0.3)

        axes[8].plot(epochs, map50, marker="o")
        axes[8].set_ylim(0.0, 1.0)
        axes[8].set_title("metrics/mAP50(B)")
        axes[8].set_xlabel("epoch")
        axes[8].grid(True, alpha=0.3)

        axes[9].plot(epochs, map5095, marker="o")
        axes[9].set_ylim(0.0, 1.0)
        axes[9].set_title("metrics/mAP50-95(B)")
        axes[9].set_xlabel("epoch")
        axes[9].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_root / "results.png", dpi=180)
        plt.close(fig)

    def _plot_box_curves(self, trainer: "Trainer") -> None:
        if not self._rows:
            return
        output_root = Path(trainer.app_config.train.output_dir)
        epochs = [row["epoch"] for row in self._rows]
        precision = [row["metrics/precision(B)"] for row in self._rows]
        recall = [row["metrics/recall(B)"] for row in self._rows]
        f1 = [row["metrics/F1(B)"] for row in self._rows]

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, precision, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("epoch")
        plt.ylabel("precision")
        plt.title("BoxP_curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "BoxP_curve.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, recall, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("epoch")
        plt.ylabel("recall")
        plt.title("BoxR_curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "BoxR_curve.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, f1, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("epoch")
        plt.ylabel("f1")
        plt.title("BoxF1_curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "BoxF1_curve.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, marker="o")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("BoxPR_curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "BoxPR_curve.png", dpi=180)
        plt.close()

    def _plot_bbox_metrics(self, trainer: "Trainer") -> None:
        if not self._rows:
            return
        output_root = Path(trainer.app_config.train.output_dir)
        epochs = [row["epoch"] for row in self._rows]
        map_all = [row["metrics/mAP50-95(B)"] for row in self._rows]
        map_50 = [row["metrics/mAP50(B)"] for row in self._rows]
        map_75 = [row["val/bbox_map_75"] for row in self._rows]
        mar_100 = [row["val/bbox_mar_100"] for row in self._rows]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, map_all, marker="o", label="bbox_map")
        plt.plot(epochs, map_50, marker="o", label="bbox_map_50")
        plt.plot(epochs, map_75, marker="o", label="bbox_map_75")
        plt.plot(epochs, mar_100, marker="o", label="bbox_mar_100")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("bbox_metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_root / "bbox_metrics.png", dpi=180)
        plt.close()
