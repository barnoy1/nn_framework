from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config_service import load_json
from .geometry import polygon_to_rle, rectangle_to_polygon, resolve_image_path


def build_coco_for_split(split_dir: Path, ann_subdir: str, img_subdir: str, logger, data_cfg: dict):
    ann_dir = split_dir / ann_subdir
    img_dir = split_dir / img_subdir
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No JSON annotation files found under {ann_dir}")

    images: List[dict] = []
    annotations: List[dict] = []
    categories_by_output_class_id: Dict[int, dict] = {}
    source_to_contiguous_class_id: Dict[int, int] = {}

    label2classid: Dict[int, str] = data_cfg["label2classid"]
    mapping: Dict[int, int] = data_cfg["mapping"]
    num_classes: int = data_cfg["num_classes"]

    image_id = 1
    annotation_id = 1
    remap_stats = {
        "split": split_dir.name,
        "source_raw_to_contiguous": {},
        "source_contiguous_to_name": {},
        "source_raw_to_target": {},
        "source_contiguous_to_target": {},
        "source_target_counts": {},
        "target_counts": {},
        "skipped_non_rectangle": 0,
        "skipped_out_of_range": 0,
        "skipped_invalid_polygon": 0,
    }

    for ann_file in ann_files:
        sample = load_json(ann_file)
        size = sample.get("size", {})
        width = int(size.get("width", 0))
        height = int(size.get("height", 0))
        if width <= 0 or height <= 0:
            logger.warning("Skipping {}: invalid size width={}, height={}", ann_file.name, width, height)
            continue

        image_path = resolve_image_path(img_dir, ann_file.stem)
        if not image_path.exists():
            logger.warning("Image file not found for annotation {}", ann_file.name)
            continue

        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        for obj in sample.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                remap_stats["skipped_non_rectangle"] += 1
                continue

            source_class_id = int(obj.get("classId"))
            class_title = str(obj.get("classTitle", f"class_{source_class_id}"))
            if source_class_id not in source_to_contiguous_class_id:
                source_to_contiguous_class_id[source_class_id] = len(source_to_contiguous_class_id)

            source_contiguous_id = source_to_contiguous_class_id[source_class_id]
            remap_stats["source_raw_to_contiguous"][str(source_class_id)] = source_contiguous_id
            remap_stats["source_contiguous_to_name"][str(source_contiguous_id)] = class_title

            class_id = int(mapping.get(source_contiguous_id, source_contiguous_id))
            if class_id < 0 or class_id >= num_classes:
                remap_stats["skipped_out_of_range"] += 1
                continue

            remap_stats["source_raw_to_target"][str(source_class_id)] = class_id
            remap_stats["source_contiguous_to_target"][str(source_contiguous_id)] = class_id
            pair_key = f"{source_contiguous_id}->{class_id}"
            remap_stats["source_target_counts"][pair_key] = remap_stats["source_target_counts"].get(pair_key, 0) + 1
            remap_stats["target_counts"][str(class_id)] = remap_stats["target_counts"].get(str(class_id), 0) + 1

            categories_by_output_class_id[class_id] = {
                "id": class_id,
                "name": label2classid.get(class_id, class_title),
                "supercategory": "none",
            }

            try:
                polygon = rectangle_to_polygon(obj.get("points", {}).get("exterior", []))
            except ValueError:
                remap_stats["skipped_invalid_polygon"] += 1
                continue

            x_coords = polygon[0::2]
            y_coords = polygon[1::2]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox_w, bbox_h = x_max - x_min, y_max - y_min

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": float(bbox_w * bbox_h),
                    "iscrowd": 0,
                    "segmentation": polygon_to_rle(polygon, height, width),
                }
            )
            annotation_id += 1

        image_id += 1

    categories = [categories_by_output_class_id[class_id] for class_id in sorted(categories_by_output_class_id.keys())]
    remap_stats["num_images"] = len(images)
    remap_stats["num_annotations"] = len(annotations)
    remap_stats["num_categories"] = len(categories)
    return {"images": images, "annotations": annotations, "categories": categories}, remap_stats


def save_coco_json(coco_data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(coco_data, file, indent=2)
