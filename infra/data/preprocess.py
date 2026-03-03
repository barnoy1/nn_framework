from __future__ import annotations

from typing import Any, Mapping, Optional

import torchvision.transforms as T


def infer_resize_size_from_loader(loader_cfg: Mapping[str, Any] | None, default: int = 640) -> int:
    if not isinstance(loader_cfg, Mapping):
        return default
    dataset_cfg = loader_cfg.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        return default
    transforms_cfg = dataset_cfg.get("transforms")
    if not isinstance(transforms_cfg, Mapping):
        return default
    ops = transforms_cfg.get("ops")
    if not isinstance(ops, list):
        return default

    for op in ops:
        if not isinstance(op, Mapping):
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


def build_image_preprocess_from_loader(
    loader_cfg: Mapping[str, Any] | None,
    logger: Optional[Any] = None,
    default_size: int = 640,
) -> T.Compose:
    ops = []

    dataset_cfg = loader_cfg.get("dataset") if isinstance(loader_cfg, Mapping) else None
    transforms_cfg = dataset_cfg.get("transforms") if isinstance(dataset_cfg, Mapping) else None
    configured_ops = transforms_cfg.get("ops") if isinstance(transforms_cfg, Mapping) else None

    if isinstance(configured_ops, list):
        for op in configured_ops:
            if not isinstance(op, Mapping):
                continue
            op_type = str(op.get("type", "")).strip().lower()

            if op_type == "resize":
                size = op.get("size")
                if isinstance(size, list) and len(size) >= 2:
                    height = int(size[0])
                    width = int(size[1])
                else:
                    height = int(size)
                    width = int(size)
                ops.append(T.Resize((height, width)))
                continue

            if op_type == "convertpilimage":
                continue

            if op_type == "normalize":
                mean = op.get("mean")
                std = op.get("std")
                if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
                    ops.append(T.Normalize(mean=list(mean), std=list(std)))
                else:
                    if logger is not None:
                        logger.warning("Invalid Normalize config in dataloader transforms; skipping")
                continue

            if logger is not None:
                logger.warning("Unsupported dataloader transform '{}' for image preprocessing; skipping.", op.get("type"))

    if not ops:
        fallback_size = infer_resize_size_from_loader(loader_cfg, default=default_size)
        ops.append(T.Resize((fallback_size, fallback_size)))

    ops.append(T.ToTensor())
    return T.Compose(ops)
