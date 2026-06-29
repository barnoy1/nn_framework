from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO


def build_eval_samples(
    val_sets, label_mapping: Dict[int, int], load_masks: bool = False
) -> List[Dict]:
    samples: List[Dict] = []
    normalized_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}

    for dataset_pair in val_sets:
        coco = COCO(str(dataset_pair.ann_file))
        dataset_name = str(Path(dataset_pair.ann_file).resolve().parent.name)
        image_ids = sorted(coco.getImgIds())
        categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda cat: cat["id"])
        category_id_to_contiguous = {
            cat["id"]: idx for idx, cat in enumerate(categories)
        }

        for image_id in image_ids:
            image_meta = coco.loadImgs([image_id])[0]
            image_path = Path(dataset_pair.img_dir) / image_meta["file_name"]
            width = int(image_meta["width"])
            height = int(image_meta["height"])

            ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            annotations = coco.loadAnns(ann_ids)

            boxes_xyxy: List[List[float]] = []
            labels: List[int] = []
            mask_list: List["np.ndarray"] = []
            for ann in annotations:
                if ann.get("iscrowd", 0) == 1:
                    continue
                category_id = ann.get("category_id")
                if category_id not in category_id_to_contiguous:
                    continue

                x, y, w, h = ann["bbox"]
                x1 = float(max(0.0, x))
                y1 = float(max(0.0, y))
                x2 = float(min(width, x + w))
                y2 = float(min(height, y + h))
                if x2 <= x1 or y2 <= y1:
                    continue

                contiguous_label = int(category_id_to_contiguous[category_id])
                mapped_label = int(
                    normalized_mapping.get(contiguous_label, contiguous_label)
                )

                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(mapped_label)
                if load_masks:
                    mask_list.append(coco.annToMask(ann).astype(bool))

            if boxes_xyxy:
                gt_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
                gt_labels = torch.tensor(labels, dtype=torch.long)
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros((0,), dtype=torch.long)

            sample = {
                "dataset_name": dataset_name,
                "image_id": int(image_id),
                "image_path": image_path,
                "file_name": str(image_meta["file_name"]),
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
            }
            if load_masks:
                if mask_list:
                    gt_masks = torch.from_numpy(np.stack(mask_list)).bool()
                else:
                    gt_masks = torch.zeros((0, height, width), dtype=torch.bool)
                sample["gt_masks"] = gt_masks
            samples.append(sample)

    return samples
