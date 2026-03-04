from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def write_metrics_json(eval_vis_dir: Path, metrics: Dict[str, float], logger) -> None:
    metrics_path = eval_vis_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump({key: float(value) for key, value in metrics.items()}, file, indent=2)
    logger.info("Saved eval metrics JSON to {}", metrics_path)


def populate_confusion_diagnostics(
    diagnostics: Optional[Dict[str, object]],
    confusion_events,
    class_id_to_name: Dict[int, str],
) -> None:
    if diagnostics is None:
        return

    class_ids = set(int(class_id) for class_id in class_id_to_name.keys())
    for gt_label, pred_label in confusion_events:
        if gt_label is not None:
            class_ids.add(int(gt_label))
        if pred_label is not None:
            class_ids.add(int(pred_label))
    max_class_id = max(class_ids) if class_ids else -1
    background_id = max_class_id + 1
    confusion_matrix = np.zeros((background_id + 1, background_id + 1), dtype=np.int64)
    for gt_label, pred_label in confusion_events:
        gt_index = background_id if gt_label is None else int(gt_label)
        pred_index = background_id if pred_label is None else int(pred_label)
        confusion_matrix[gt_index, pred_index] += 1

    confusion_labels = [class_id_to_name.get(class_id, str(class_id)) for class_id in range(background_id)]
    confusion_labels.append("background")
    diagnostics["confusion_matrix"] = confusion_matrix
    diagnostics["confusion_labels"] = confusion_labels
