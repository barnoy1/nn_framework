from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_detection_scores(matrix: np.ndarray | None) -> Dict[str, float]:
    if matrix is None or matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    class_count = int(matrix.shape[0] - 1)
    core = matrix[:class_count, :class_count]
    tp = float(np.trace(core))
    pred_non_bg = float(matrix[:, :class_count].sum())
    gt_non_bg = float(matrix[:class_count, :].sum())
    fp = max(0.0, pred_non_bg - tp)
    fn = max(0.0, gt_non_bg - tp)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0.0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0.0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    accuracy = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0.0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _save_confusion_matrix_plot(output_root: Path, matrix: np.ndarray, labels: List[str]) -> None:
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


def _save_confusion_matrix_normalized_plot(output_root: Path, matrix: np.ndarray, labels: List[str]) -> None:
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


def _write_results_csv(output_root: Path, metrics: Dict[str, float], scores: Dict[str, float]) -> Path:
    results_csv = output_root / "results.csv"
    fieldnames = [
        "epoch",
        "time",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/F1(B)",
        "metrics/accuracy(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/bbox_map_75",
        "val/bbox_mar_100",
    ]
    row = {
        "epoch": 1.0,
        "time": 1.0,
        "metrics/precision(B)": float(scores.get("precision", 0.0)),
        "metrics/recall(B)": float(scores.get("recall", 0.0)),
        "metrics/F1(B)": float(scores.get("f1", 0.0)),
        "metrics/accuracy(B)": float(scores.get("accuracy", 0.0)),
        "metrics/mAP50(B)": float(metrics.get("bbox_map_50", 0.0)),
        "metrics/mAP50-95(B)": float(metrics.get("bbox_map", metrics.get("map", 0.0))),
        "val/bbox_map_75": float(metrics.get("bbox_map_75", 0.0)),
        "val/bbox_mar_100": float(metrics.get("bbox_mar_100", 0.0)),
    }
    with results_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return results_csv


def _plot_bbox_metrics(output_root: Path, metrics: Dict[str, float]) -> None:
    epochs = [1.0]
    map_all = [float(metrics.get("bbox_map", metrics.get("map", 0.0)))]
    map_50 = [float(metrics.get("bbox_map_50", 0.0))]
    map_75 = [float(metrics.get("bbox_map_75", 0.0))]
    mar_100 = [float(metrics.get("bbox_mar_100", 0.0))]

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


def _plot_box_curves(output_root: Path, scores: Dict[str, float]) -> None:
    epochs = [1.0]
    precision = [float(scores.get("precision", 0.0))]
    recall = [float(scores.get("recall", 0.0))]
    f1 = [float(scores.get("f1", 0.0))]

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


def _plot_results(output_root: Path, metrics: Dict[str, float], scores: Dict[str, float]) -> None:
    epochs = [1.0]
    precision = [float(scores.get("precision", 0.0))]
    recall = [float(scores.get("recall", 0.0))]
    map50 = [float(metrics.get("bbox_map_50", 0.0))]
    map5095 = [float(metrics.get("bbox_map", metrics.get("map", 0.0)))]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    axes[0].plot(epochs, precision, marker="o")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("metrics/precision(B)")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, recall, marker="o")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("metrics/recall(B)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, map50, marker="o")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("metrics/mAP50(B)")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, map5095, marker="o")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("metrics/mAP50-95(B)")
    axes[3].set_xlabel("epoch")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_root / "results.png", dpi=180)
    plt.close(fig)


def generate_eval_model_metrics_bundle(
    *,
    output_root: Path,
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray | None,
    confusion_labels: List[str],
) -> Dict[str, float]:
    output_root.mkdir(parents=True, exist_ok=True)
    scores = compute_detection_scores(confusion_matrix)
    _write_results_csv(output_root=output_root, metrics=metrics, scores=scores)
    _plot_bbox_metrics(output_root=output_root, metrics=metrics)
    _plot_box_curves(output_root=output_root, scores=scores)
    _plot_results(output_root=output_root, metrics=metrics, scores=scores)
    if confusion_matrix is not None and confusion_labels:
        _save_confusion_matrix_plot(output_root=output_root, matrix=confusion_matrix, labels=confusion_labels)
        _save_confusion_matrix_normalized_plot(output_root=output_root, matrix=confusion_matrix, labels=confusion_labels)
    return scores


def log_eval_model_metrics_bundle(
    *,
    output_root: Path,
    vis_logger,
    step: int,
    metrics: Dict[str, float],
    confusion_scores: Dict[str, float],
    confusion_labels: List[str],
) -> None:
    vis_logger.log_metrics(
        metrics={f"eval/confusion/{key}": float(value) for key, value in confusion_scores.items()},
        step=step,
    )
    if confusion_labels:
        vis_logger.log_text(
            tag="eval/confusion_labels",
            text=json.dumps(confusion_labels, ensure_ascii=False),
            step=step,
        )

    artifact_candidates = [
        output_root / "results.csv",
        output_root / "results.png",
        output_root / "bbox_metrics.png",
        output_root / "BoxP_curve.png",
        output_root / "BoxR_curve.png",
        output_root / "BoxF1_curve.png",
        output_root / "BoxPR_curve.png",
        output_root / "confusion_matrix.png",
        output_root / "confusion_matrix_normalized.png",
    ]
    for artifact_path in artifact_candidates:
        if artifact_path.exists():
            vis_logger.log_artifact(file_path=artifact_path, artifact_path="model-metrics")

    for val_batch_path in sorted(output_root.glob("val_batch*.jpg")):
        vis_logger.log_artifact(file_path=val_batch_path, artifact_path="model-metrics")

    dataset_labels = output_root / "dataset" / "labels.png"
    if dataset_labels.exists():
        vis_logger.log_artifact(file_path=dataset_labels, artifact_path="dataset")
