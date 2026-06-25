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

    label_id_to_name = {
        int(label_id): str(name) for label_id, name in class_id_to_name.items()
    }
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


def _stack_triplet(labels, boxes, scores) -> list:
    if labels.ndim == 1:
        labels, boxes, scores = labels[None], boxes[None], scores[None]
    return [
        {"labels": li, "boxes": bi, "scores": si}
        for li, bi, si in zip(labels, boxes, scores)
    ]


def _as_prediction_dicts(obj) -> list:
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        return _stack_triplet(*obj)
    if isinstance(obj, dict) and {"labels", "boxes", "scores"}.issubset(obj):
        return _stack_triplet(obj["labels"], obj["boxes"], obj["scores"])
    raise TypeError(f"Unsupported postprocessor output: {type(obj)}")


def to_canonical_predictions(outputs, postprocessor, orig_sizes, *, iou_types=()):
    """Single canonical shape `{labels, boxes, scores, masks?}` per image.

    `masks` is kept iff `segm in iou_types` and the postprocessor produced it.
    """
    want_masks = "segm" in set(iou_types or ())
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        raw = outputs
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        raw = _stack_triplet(*outputs)
    else:
        raw = _as_prediction_dicts(postprocessor(outputs, orig_sizes))

    canonical = []
    for item in raw:
        result = {
            "labels": item["labels"],
            "boxes": item["boxes"],
            "scores": item["scores"],
        }
        if want_masks and item.get("masks") is not None:
            result["masks"] = item["masks"]
        canonical.append(result)
    return canonical


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
        mapped_flat = [
            label_id_remap.get(int(label), int(label)) for label in flat_labels
        ]
        mapped = torch.tensor(
            mapped_flat, dtype=labels.dtype, device=labels.device
        ).reshape_as(labels)

        normalized = dict(result)
        normalized["labels"] = mapped
        normalized_results.append(normalized)

    return normalized_results


if __name__ == "__main__":
    # ponytail: smallest shape check — detection omits masks, segm keeps them.
    import torch as _t

    item = {
        "labels": _t.tensor([1]),
        "boxes": _t.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "scores": _t.tensor([0.9]),
        "masks": _t.zeros(1, 2, 2),
    }
    det = to_canonical_predictions([item], None, None, iou_types=("bbox",))[0]
    seg = to_canonical_predictions([item], None, None, iou_types=("bbox", "segm"))[0]
    assert set(det) == {"labels", "boxes", "scores"}, det.keys()
    assert "masks" in seg and set(seg) == {"labels", "boxes", "scores", "masks"}
    print("prediction shape self-check OK")
