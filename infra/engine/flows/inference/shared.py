from __future__ import annotations

from pathlib import Path
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_resize_size_from_loader(loader_cfg: dict | None, default: int = 640) -> int:
    if not isinstance(loader_cfg, dict):
        return default
    dataset_cfg = loader_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return default
    transforms_cfg = dataset_cfg.get("transforms")
    if not isinstance(transforms_cfg, dict):
        return default
    ops = transforms_cfg.get("ops")
    if not isinstance(ops, list):
        return default

    for op in ops:
        if not isinstance(op, dict):
            continue
        if str(op.get("type", "")).lower() != "resize":
            continue
        size = op.get("size")
        if isinstance(size, list) and len(size) >= 2:
            try:
                return int(size[0])
            except (TypeError, ValueError):
                return default
        try:
            return int(size)
        except (TypeError, ValueError):
            return default

    return default


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    return [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMG_EXTS]
