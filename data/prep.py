from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from nn_framework.utils.log import logger


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def rectangle_to_polygon(exterior: List[List[float]]) -> List[float]:
    if len(exterior) != 2:
        raise ValueError(f"Rectangle exterior must contain exactly 2 points, got {len(exterior)}")
    (x1, y1), (x2, y2) = exterior
    x_min, x_max = sorted([float(x1), float(x2)])
    y_min, y_max = sorted([float(y1), float(y2)])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid rectangle coordinates")
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]


def polygon_to_rle(polygon: List[float], height: int, width: int) -> Dict[str, object]:
    rles = mask_utils.frPyObjects([polygon], height, width)
    mask = mask_utils.decode(rles)
    encoded = mask_utils.encode(np.asfortranarray(mask[:, :, 0]))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return {"size": encoded["size"], "counts": encoded["counts"]}


def resolve_image_path(img_dir: Path, stem: str) -> Path:
    candidate = img_dir / stem
    if candidate.exists() and candidate.is_file():
        return candidate
    normalized_stem = Path(stem).stem
    for ext in IMAGE_EXTENSIONS:
        ext_candidate = img_dir / f"{normalized_stem}{ext}"
        if ext_candidate.exists():
            return ext_candidate
    return img_dir / f"{normalized_stem}.jpg"


def normalize_bbox(polygon: Sequence[float], width: int, height: int) -> List[float]:
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]
    x_min, x_max = max(0.0, min(x_coords)), min(float(width), max(x_coords))
    y_min, y_max = max(0.0, min(y_coords)), min(float(height), max(y_coords))
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    return [x_min, y_min, w, h]


def convert_split(split_dir: Path, ann_subdir: str = "ann", img_subdir: str = "img") -> dict:
    ann_dir = split_dir / ann_subdir
    img_dir = split_dir / img_subdir
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found under {ann_dir}")

    images: List[dict] = []
    annotations: List[dict] = []
    source_to_contiguous: Dict[int, int] = {}
    categories: Dict[int, dict] = {}

    image_id = 1
    annotation_id = 1

    for ann_file in ann_files:
        sample = load_json(ann_file)
        width = int(sample.get("size", {}).get("width", 0))
        height = int(sample.get("size", {}).get("height", 0))
        if width <= 0 or height <= 0:
            continue

        image_path = resolve_image_path(img_dir, ann_file.stem)
        if not image_path.exists():
            continue

        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        for obj in sample.get("objects", []):
            geometry_type = obj.get("geometryType")
            if geometry_type == "rectangle":
                polygon = rectangle_to_polygon(obj.get("points", {}).get("exterior", []))
            elif geometry_type == "polygon":
                exterior = obj.get("points", {}).get("exterior", [])
                polygon = [float(v) for pt in exterior for v in pt]
                if len(polygon) < 6:
                    continue
            else:
                continue

            source_class_id = int(obj.get("classId"))
            class_name = str(obj.get("classTitle", f"class_{source_class_id}"))
            if source_class_id not in source_to_contiguous:
                contiguous_id = len(source_to_contiguous)
                source_to_contiguous[source_class_id] = contiguous_id
                categories[source_class_id] = {"id": contiguous_id, "name": class_name, "supercategory": "none"}

            category_id = source_to_contiguous[source_class_id]
            bbox = normalize_bbox(polygon, width, height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            segmentation = polygon_to_rle(polygon, height, width)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )
            annotation_id += 1

        image_id += 1

    ordered_categories = [
        categories[src_id] for src_id, _ in sorted(source_to_contiguous.items(), key=lambda item: item[1])
    ]
    return {"images": images, "annotations": annotations, "categories": ordered_categories}


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    splits: Sequence[str] = ("train", "valid"),
    ann_subdir: str = "ann",
    img_subdir: str = "img",
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[Path] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        coco = convert_split(split_dir, ann_subdir=ann_subdir, img_subdir=img_subdir)
        out_path = output_dir / f"instances_{split}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(coco, file, indent=2)
        written_files.append(out_path)
    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Supervisely annotations to COCO RLE")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "valid"])
    parser.add_argument("--ann_subdir", type=str, default="ann")
    parser.add_argument("--img_subdir", type=str, default="img")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = convert_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        splits=args.splits,
        ann_subdir=args.ann_subdir,
        img_subdir=args.img_subdir,
    )
    for path in files:
        logger.info("{}", path)


if __name__ == "__main__":
    main()
