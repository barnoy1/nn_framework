from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def build_label_id_remap_from_config_and_annotations(
    *,
    remap_mscoco_category: bool,
    class_id_to_name: Dict[int, str],
    annotation_files: Iterable[str],
) -> Dict[int, int]:
    if not remap_mscoco_category:
        return {}

    if not class_id_to_name:
        return {}

    label_id_to_name = {int(label_id): str(name) for label_id, name in class_id_to_name.items()}
    name_to_label_id = {name: label_id for label_id, name in label_id_to_name.items()}
    remap: Dict[int, int] = {}

    for ann_file in annotation_files:
        try:
            payload = json.loads(Path(ann_file).read_text(encoding="utf-8"))
        except Exception:
            continue

        categories = payload.get("categories", []) if isinstance(payload, dict) else []
        if not isinstance(categories, list):
            continue

        for category in categories:
            if not isinstance(category, dict):
                continue
            category_id = category.get("id")
            category_name = category.get("name")
            if category_id is None or not isinstance(category_name, str):
                continue
            if category_name not in name_to_label_id:
                continue
            remap[int(category_id)] = int(name_to_label_id[category_name])

    return remap


def to_result_list(outputs, postprocessor, orig_sizes):
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs

    if isinstance(outputs, dict):
        processed = postprocessor(outputs, orig_sizes)
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        labels, boxes, scores = outputs
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]
    else:
        processed = postprocessor(outputs, orig_sizes)

    if isinstance(processed, list):
        return processed

    if isinstance(processed, (tuple, list)) and len(processed) == 3:
        labels, boxes, scores = processed
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    if isinstance(processed, dict) and {"labels", "boxes", "scores"}.issubset(set(processed.keys())):
        labels = processed["labels"]
        boxes = processed["boxes"]
        scores = processed["scores"]
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
        return [
            {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    raise TypeError(f"Unsupported output format: {type(processed)}")


def normalize_prediction_labels_for_metrics(
    results,
    *,
    label_id_remap: Optional[Dict[int, int]] = None,
):
    if not label_id_remap:
        return results

    normalized_results = []
    for result in results:
        labels = result.get("labels")
        if not isinstance(labels, torch.Tensor):
            normalized_results.append(result)
            continue

        flat_labels = labels.detach().to("cpu").reshape(-1).tolist()
        mapped_flat = [label_id_remap.get(int(label), int(label)) for label in flat_labels]
        mapped = torch.tensor(mapped_flat, dtype=labels.dtype, device=labels.device).reshape_as(labels)

        normalized = dict(result)
        normalized["labels"] = mapped
        normalized_results.append(normalized)

    return normalized_results
