from __future__ import annotations

from pathlib import Path

from .config_service import load_json
from .geometry import bbox_to_polygon, polygon_to_rle


def build_coco_from_coco_split(split_dir: Path, logger, data_cfg=None):
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")

    payload = load_json(ann_path)
    source_categories = payload.get("categories", [])
    source_images = payload.get("images", [])
    source_annotations = payload.get("annotations", [])

    label2classid = data_cfg["label2classid"] if data_cfg else {}
    category_items = sorted(
        [category for category in source_categories if int(category.get("id", -1)) != 0],
        key=lambda category: int(category["id"]),
    )
    source_to_contiguous = {
        int(category["id"]): index for index, category in enumerate(category_items)
    }
    categories = [
        {
            "id": index,
            "name": label2classid.get(index, str(category.get("name", f"class_{index}"))),
            "supercategory": "none",
        }
        for index, category in enumerate(category_items)
    ]

    image_id_map = {}
    images = []
    for new_id, image in enumerate(
        sorted(source_images, key=lambda current: int(current.get("id", -1))), start=1
    ):
        source_image_id = int(image.get("id", -1))
        image_id_map[source_image_id] = new_id
        images.append(
            {
                "id": new_id,
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
    image_meta_by_new_id = {image["id"]: image for image in images}

    remap_stats = {
        "split": split_dir.name,
        "source_raw_to_contiguous": {
            str(source_id): contiguous for source_id, contiguous in source_to_contiguous.items()
        },
        "source_contiguous_to_name": {
            str(category["id"]): category["name"] for category in categories
        },
        "source_raw_to_target": {
            str(source_id): contiguous for source_id, contiguous in source_to_contiguous.items()
        },
        "source_contiguous_to_target": {
            str(contiguous): contiguous for contiguous in range(len(categories))
        },
        "target_class_to_name": {
            str(category["id"]): category["name"] for category in categories
        },
        "source_target_counts": {},
        "target_counts": {},
        "skipped_non_rectangle": 0,
        "skipped_out_of_range": 0,
        "skipped_invalid_polygon": 0,
        "skipped_dropped_category": 0,
        "num_images": len(images),
        "num_annotations": 0,
        "num_categories": len(categories),
    }

    annotations = []
    next_annotation_id = 1
    for source_annotation in source_annotations:
        source_category_id = int(source_annotation.get("category_id", -1))
        if source_category_id == 0:
            remap_stats["skipped_dropped_category"] += 1
            continue

        target_category_id = source_to_contiguous.get(source_category_id)
        if target_category_id is None:
            remap_stats["skipped_out_of_range"] += 1
            continue

        source_image_id = int(source_annotation.get("image_id", -1))
        new_image_id = image_id_map.get(source_image_id)
        if new_image_id is None:
            logger.warning(
                "Skipping annotation {}: image_id {} not found in images",
                source_annotation.get("id"),
                source_image_id,
            )
            continue

        bbox = source_annotation.get("bbox", [])
        try:
            polygon = bbox_to_polygon(bbox)
        except (TypeError, ValueError):
            remap_stats["skipped_invalid_polygon"] += 1
            continue

        bbox_float = [float(value) for value in bbox]
        image_meta = image_meta_by_new_id[new_image_id]
        remap_stats["source_target_counts"][f"{target_category_id}->{target_category_id}"] = (
            remap_stats["source_target_counts"].get(
                f"{target_category_id}->{target_category_id}", 0
            )
            + 1
        )
        remap_stats["target_counts"][str(target_category_id)] = (
            remap_stats["target_counts"].get(str(target_category_id), 0) + 1
        )

        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": new_image_id,
                "category_id": target_category_id,
                "bbox": bbox_float,
                "area": float(bbox_float[2] * bbox_float[3]),
                "iscrowd": 0,
                "segmentation": polygon_to_rle(
                    polygon, int(image_meta["height"]), int(image_meta["width"])
                ),
            }
        )
        next_annotation_id += 1

    remap_stats["num_annotations"] = len(annotations)
    return {"images": images, "annotations": annotations, "categories": categories}, remap_stats


if __name__ == "__main__":  # pragma: no cover - ponytail: tiny self-check for bbox->RLE path
    import json
    import tempfile

    from pycocotools import mask as mask_utils

    class _Logger:
        @staticmethod
        def warning(*_args, **_kwargs):
            return None

    with tempfile.TemporaryDirectory() as tmp:
        split_dir = Path(tmp) / "train"
        split_dir.mkdir(parents=True, exist_ok=True)
        sample = {
            "images": [{"id": 10, "file_name": "a.jpg", "width": 100, "height": 80}],
            "categories": [
                {"id": 0, "name": "ships"},
                {"id": 1, "name": "class_a"},
                {"id": 2, "name": "class_b"},
            ],
            "annotations": [{"id": 99, "image_id": 10, "category_id": 2, "bbox": [10, 20, 30, 15]}],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(sample), encoding="utf-8"
        )
        coco_data, stats = build_coco_from_coco_split(split_dir, _Logger())
        assert list(coco_data.keys()) == ["images", "annotations", "categories"]
        assert stats["source_raw_to_contiguous"] == {"1": 0, "2": 1}
        assert coco_data["annotations"][0]["category_id"] == 1
        decoded = mask_utils.decode(coco_data["annotations"][0]["segmentation"])
        assert int(decoded.sum()) == int(coco_data["annotations"][0]["area"])
    print("coco_reader self-check OK")
