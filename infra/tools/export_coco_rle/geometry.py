from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from pycocotools import mask as mask_utils

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def rectangle_to_polygon(exterior: List[List[float]]) -> List[float]:
    if len(exterior) != 2:
        raise ValueError(
            f"Rectangle exterior must contain exactly 2 points, got {len(exterior)}"
        )
    (x1, y1), (x2, y2) = exterior
    x_min, x_max = sorted([float(x1), float(x2)])
    y_min, y_max = sorted([float(y1), float(y2)])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid rectangle coordinates: {(x1, y1)} to {(x2, y2)}")
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]


def bbox_to_polygon(bbox: List[float]) -> List[float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox [x, y, w, h], got {bbox}")
    x_min, y_min, width, height = [float(value) for value in bbox]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox size: width={width}, height={height}")
    return [x_min, y_min, x_min + width, y_min, x_min + width, y_min + height, x_min, y_min + height]


def polygon_to_rle(polygon: List[float], height: int, width: int) -> Dict[str, object]:
    rles = mask_utils.frPyObjects([polygon], height, width)
    mask = mask_utils.decode(rles)
    encoded = mask_utils.encode(np.asfortranarray(mask[:, :, 0]))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return {"size": encoded["size"], "counts": encoded["counts"]}


def bbox_to_polygon(bbox: List[float]) -> List[float]:
    x, y, w, h = (float(value) for value in bbox)
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def resolve_image_path(img_dir: Path, stem: str) -> Path:
    direct_candidate = img_dir / stem
    if direct_candidate.exists() and direct_candidate.is_file():
        return direct_candidate

    normalized_stem = stem
    stem_suffix = Path(stem).suffix.lower()
    if stem_suffix in IMAGE_EXTENSIONS:
        normalized_stem = Path(stem).stem

    for ext in IMAGE_EXTENSIONS:
        candidate = img_dir / f"{normalized_stem}{ext}"
        if candidate.exists():
            return candidate

    fallback = img_dir / stem
    if fallback.suffix.lower() not in IMAGE_EXTENSIONS:
        fallback = img_dir / f"{normalized_stem}.jpg"
    return fallback
