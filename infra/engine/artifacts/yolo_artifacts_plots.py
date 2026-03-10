from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
    labels = [
        train_names.get(class_id, val_names.get(class_id, str(class_id)))
        for class_id in class_ids
    ]
    x = np.arange(len(class_ids), dtype=np.float32)
    width = 0.4
    plt.figure(figsize=(max(10, len(class_ids) * 0.35), 7))
    plt.bar(x - width / 2, train_values, width=width, label="train")
    plt.bar(x + width / 2, val_values, width=width, label="val")
    plt.xticks(
        x,
        [f"{cid}:{name}" for cid, name in zip(class_ids, labels)],
        rotation=45,
        ha="right",
    )
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
        return 0.0 if denominator <= 0.0 else float(numerator / denominator)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def save_confusion_matrix_plots(
    *, output_root: Path, matrix: np.ndarray, labels: List[str]
) -> None:
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


def render_training_artifact_plots(
    *, output_root: Path, rows: List[Dict[str, float]]
) -> None:
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    metrics_fig, metrics_axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metric_series = [
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)", "metrics/recall(B)"),
        ("metrics/mAP50(B)", "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
    ]
    for axis, (field, title) in zip(metrics_axes, metric_series):
        axis.plot(epochs, [row[field] for row in rows], marker="o")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    metrics_fig.tight_layout()
    metrics_fig.savefig(output_root / "results.png", dpi=180)
    plt.close(metrics_fig)

    for prefix in ("train", "val"):
        common_component_fields = sorted(
            {
                field
                for row in rows
                for field in row.keys()
                if field.startswith(f"{prefix}/")
                and field.startswith(f"{prefix}/common_")
            }
        )
        common_component_fields = [
            field
            for field in common_component_fields
            if all(field in row for row in rows)
        ]

        plt.figure(figsize=(9, 5))
        for field in common_component_fields:
            label = field.removeprefix(f"{prefix}/")
            if label.startswith("common_"):
                label = label.removeprefix("common_")
            plt.plot(
                epochs,
                [row[field] for row in rows],
                marker="o",
                label=label,
            )
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{prefix}_common_loss_components")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="center right", bbox_to_anchor=(-0.18, 0.5), fontsize=8)
        plt.tight_layout(rect=(0.18, 0.0, 1.0, 1.0))
        plt.savefig(output_root / f"{prefix}_common_loss_components.png", dpi=180)
        plt.close()

        model_fields = sorted(
            field
            for field in rows[0].keys()
            if field.startswith(f"{prefix}/criterion/")
        )
        present_model_fields = [
            field for field in model_fields if all(field in row for row in rows)
        ]
        if present_model_fields:
            plt.figure(figsize=(9, 5))
            for field in present_model_fields:
                label = field.split("/", 2)[-1].removesuffix("_loss")
                plt.plot(epochs, [row[field] for row in rows], marker="o", label=label)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{prefix}_concreate_loss_components")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="center right", bbox_to_anchor=(-0.18, 0.5), fontsize=8)
            plt.tight_layout(rect=(0.18, 0.0, 1.0, 1.0))
            plt.savefig(
                output_root / f"{prefix}_concreate_loss_components.png", dpi=180
            )
            plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train/total_loss"] for row in rows], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train_total_loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "train_total_loss.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["val/total_loss"] for row in rows], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("val_total_loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "val_total_loss.png", dpi=180)
    plt.close()
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
