from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

RESULTS_FIELDNAMES = [
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
    "val_map",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    "metrics/F1(B)",
    "metrics/accuracy(B)",
    "val/bbox_map_75",
    "val/bbox_mar_100",
]


def build_epoch_row(*, epoch: int, metrics: Dict[str, float], optimizer_lrs: List[float], precision: float, recall: float, f1: float, accuracy: float) -> Dict[str, float]:
    lr0 = optimizer_lrs[0] if len(optimizer_lrs) > 0 else 0.0
    lr1 = optimizer_lrs[1] if len(optimizer_lrs) > 1 else lr0
    lr2 = optimizer_lrs[2] if len(optimizer_lrs) > 2 else lr1

    return {
        "epoch": float(epoch + 1),
        "time": float(epoch + 1),
        "train/box_loss": float(metrics.get("train_loss", 0.0)),
        "train/cls_loss": float(metrics.get("train_cls_loss", 0.0)),
        "train/dfl_loss": float(metrics.get("train_dfl_loss", 0.0)),
        "train/custom_loss": float(metrics.get("train_custom_loss", 0.0)),
        "metrics/precision(B)": float(precision),
        "metrics/recall(B)": float(recall),
        "metrics/F1(B)": float(f1),
        "metrics/accuracy(B)": float(accuracy),
        "metrics/mAP50(B)": float(metrics.get("val_bbox_map_50", 0.0)),
        "metrics/mAP50-95(B)": float(metrics.get("val_bbox_map", metrics.get("val_map", 0.0))),
        "val/box_loss": float(metrics.get("val_box_loss", 0.0)),
        "val/cls_loss": float(metrics.get("val_cls_loss", 0.0)),
        "val/dfl_loss": float(metrics.get("val_dfl_loss", 0.0)),
        "val/custom_loss": float(metrics.get("val_custom_loss", 0.0)),
        "val_map": float(metrics.get("val_map", metrics.get("val_bbox_map", 0.0))),
        "lr/pg0": float(lr0),
        "lr/pg1": float(lr1),
        "lr/pg2": float(lr2),
        "val/bbox_map_75": float(metrics.get("val_bbox_map_75", 0.0)),
        "val/bbox_mar_100": float(metrics.get("val_bbox_mar_100", 0.0)),
    }


def write_results_csv(csv_path: Path, rows: List[Dict[str, float]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
