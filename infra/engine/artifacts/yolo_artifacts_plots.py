from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_labels_plot(
    *,
    output_root: Path,
    train_counts: Dict[int, int],
    train_names: Dict[int, str],
    val_counts: Dict[int, int],
    val_names: Dict[int, str],
) -> None:
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


def compute_detection_scores(matrix: np.ndarray) -> Dict[str, float]:
    if matrix is None or matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    class_count = int(matrix.shape[0] - 1)
    core = matrix[:class_count, :class_count]
    tp = float(np.trace(core))
    pred_non_bg = float(matrix[:, :class_count].sum())
    gt_non_bg = float(matrix[:class_count, :].sum())
    fp = max(0.0, pred_non_bg - tp)
    fn = max(0.0, gt_non_bg - tp)

    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def save_confusion_matrix_plots(*, output_root: Path, matrix: np.ndarray, labels: List[str]) -> None:
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

    row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    normalized = np.divide(
        matrix.astype(np.float64),
        np.where(row_sums > 0.0, row_sums, 1.0),
        out=np.zeros_like(matrix, dtype=np.float64),
    )
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("confusion_matrix_normalized")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.tight_layout()
    plt.savefig(output_root / "confusion_matrix_normalized.png", dpi=180)
    plt.close()


def render_training_artifact_plots(*, output_root: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    axes = axes.flatten()

    series: List[Tuple[str, str, bool]] = [
        ("train/box_loss", "train/box_loss", False),
        ("train/cls_loss", "train/cls_loss", False),
        ("train/dfl_loss", "train/dfl_loss", False),
        ("metrics/precision(B)", "metrics/precision(B)", True),
        ("metrics/recall(B)", "metrics/recall(B)", True),
        ("val/box_loss", "val/box_loss", False),
        ("val/cls_loss", "val/cls_loss", False),
        ("val/dfl_loss", "val/dfl_loss", False),
        ("metrics/mAP50(B)", "metrics/mAP50(B)", True),
        ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)", True),
    ]

    for axis, (field, title, bounded) in zip(axes, series):
        values = [row[field] for row in rows]
        axis.plot(epochs, values, marker="o")
        if bounded:
            axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_root / "results.png", dpi=180)
    plt.close(fig)

    precision = [row["metrics/precision(B)"] for row in rows]
    recall = [row["metrics/recall(B)"] for row in rows]
    f1 = [row["metrics/F1(B)"] for row in rows]

    for values, name, ylabel in [
        (precision, "BoxP_curve.png", "precision"),
        (recall, "BoxR_curve.png", "recall"),
        (f1, "BoxF1_curve.png", "f1"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, values, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(name.removesuffix(".png"))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / name, dpi=180)
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

    map_all = [row["metrics/mAP50-95(B)"] for row in rows]
    map_50 = [row["metrics/mAP50(B)"] for row in rows]
    map_75 = [row["val/bbox_map_75"] for row in rows]
    mar_100 = [row["val/bbox_mar_100"] for row in rows]

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
