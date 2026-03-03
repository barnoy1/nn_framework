from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import albumentations as A
import cv2
import numpy as np


@dataclass
class TransformResult:
    image: np.ndarray
    bboxes: List[List[float]]
    class_labels: List[int]
    masks: Optional[List[np.ndarray]] = None


class DynamicAlbumentations:
    def __init__(
        self,
        use_masks: bool = False,
        image_size: int = 640,
        heavy_scale_min: float = 0.5,
        heavy_scale_max: float = 1.5,
        light_scale_min: float = 0.9,
        light_scale_max: float = 1.1,
        switch_epoch_ratio: float = 0.85,
        horizontal_flip_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
    ):
        self.use_masks = use_masks
        self.image_size = image_size
        self.heavy_scale_min = heavy_scale_min
        self.heavy_scale_max = heavy_scale_max
        self.light_scale_min = light_scale_min
        self.light_scale_max = light_scale_max
        self.switch_epoch_ratio = switch_epoch_ratio
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.current_stage = "heavy"
        self._heavy = self._build_heavy_pipeline()
        self._light = self._build_light_pipeline()

    def _build_common(self) -> List[Any]:
        return [
            A.HorizontalFlip(p=self.horizontal_flip_prob),
            A.RandomBrightnessContrast(p=self.color_jitter_prob),
        ]

    def _build_heavy_pipeline(self) -> A.Compose:
        ops = [
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=(self.heavy_scale_min - 1.0, self.heavy_scale_max - 1.0), p=0.9),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomSizedBBoxSafeCrop(height=self.image_size, width=self.image_size, p=1.0),
            *self._build_common(),
        ]
        return self._compose(ops)

    def _build_light_pipeline(self) -> A.Compose:
        ops = [
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=(self.light_scale_min - 1.0, self.light_scale_max - 1.0), p=0.4),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(height=self.image_size, width=self.image_size, p=1.0),
            *self._build_common(),
        ]
        return self._compose(ops)

    def _compose(self, ops: List[Any]) -> A.Compose:
        return A.Compose(
            ops,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.05),
        )

    def update_augmentation(self, epoch: int, total_epochs: int) -> None:
        switch_epoch = int(total_epochs * self.switch_epoch_ratio)
        self.current_stage = "heavy" if epoch < switch_epoch else "light"

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        masks: Optional[List[np.ndarray]] = None,
    ) -> TransformResult:
        compose = self._heavy if self.current_stage == "heavy" else self._light
        payload: Dict[str, Any] = {"image": image, "bboxes": bboxes, "class_labels": class_labels}
        if self.use_masks and masks is not None:
            payload["masks"] = masks
        transformed = compose(**payload)
        return TransformResult(
            image=transformed["image"],
            bboxes=[list(b) for b in transformed["bboxes"]],
            class_labels=[int(v) for v in transformed["class_labels"]],
            masks=transformed.get("masks"),
        )


class EvalResizeTransform:
    def __init__(self, use_masks: bool = False, image_size: int = 640):
        self.use_masks = use_masks
        self.image_size = image_size
        self._compose = A.Compose(
            [A.Resize(height=self.image_size, width=self.image_size)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.05),
        )

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        masks: Optional[List[np.ndarray]] = None,
    ) -> TransformResult:
        payload: Dict[str, Any] = {"image": image, "bboxes": bboxes, "class_labels": class_labels}
        if self.use_masks and masks is not None:
            payload["masks"] = masks
        transformed = self._compose(**payload)
        return TransformResult(
            image=transformed["image"],
            bboxes=[list(b) for b in transformed["bboxes"]],
            class_labels=[int(v) for v in transformed["class_labels"]],
            masks=transformed.get("masks"),
        )


class ConfigurableAlbumentations:
    def __init__(self, ops: List[Any], use_masks: bool = False):
        self.use_masks = use_masks
        self._compose = A.Compose(
            ops,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.05),
        )

    def __call__(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        masks: Optional[List[np.ndarray]] = None,
    ) -> TransformResult:
        payload: Dict[str, Any] = {"image": image, "bboxes": bboxes, "class_labels": class_labels}
        if self.use_masks and masks is not None:
            payload["masks"] = masks
        transformed = self._compose(**payload)
        return TransformResult(
            image=transformed["image"],
            bboxes=[list(b) for b in transformed["bboxes"]],
            class_labels=[int(v) for v in transformed["class_labels"]],
            masks=transformed.get("masks"),
        )


def _prob(op_cfg: Mapping[str, Any], default: float = 1.0) -> float:
    raw = op_cfg.get("p", default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _extract_resize_hw(op_cfg: Mapping[str, Any], default_size: int) -> tuple[int, int]:
    size = op_cfg.get("size", default_size)
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    resolved = int(size)
    return resolved, resolved


def build_albumentations_from_loader(
    loader_cfg: Mapping[str, Any] | None,
    use_masks: bool,
    default_size: int = 640,
) -> ConfigurableAlbumentations:
    dataset_cfg = loader_cfg.get("dataset") if isinstance(loader_cfg, Mapping) else None
    transforms_cfg = dataset_cfg.get("transforms") if isinstance(dataset_cfg, Mapping) else None
    configured_ops = transforms_cfg.get("ops") if isinstance(transforms_cfg, Mapping) else None

    ops: List[Any] = []
    resize_h, resize_w = default_size, default_size

    if isinstance(configured_ops, list):
        for op in configured_ops:
            if not isinstance(op, Mapping):
                continue

            op_type = str(op.get("type", "")).strip().lower()

            if op_type == "resize":
                resize_h, resize_w = _extract_resize_hw(op, default_size=default_size)
                ops.append(A.Resize(height=resize_h, width=resize_w, p=_prob(op, 1.0)))
                continue

            if op_type == "randomhorizontalflip":
                ops.append(A.HorizontalFlip(p=_prob(op, 0.5)))
                continue

            if op_type == "randomverticalflip":
                ops.append(A.VerticalFlip(p=_prob(op, 0.5)))
                continue

            if op_type == "rotate":
                limit = op.get("limit", op.get("angle", 15))
                if isinstance(limit, list) and len(limit) >= 2:
                    limit_value = (float(limit[0]), float(limit[1]))
                else:
                    bound = float(limit)
                    limit_value = (-bound, bound)
                ops.append(
                    A.Rotate(
                        limit=limit_value,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=_prob(op, 0.5),
                    )
                )
                continue

            if op_type == "randomphotometricdistort":
                ops.append(A.ColorJitter(p=_prob(op, 0.5)))
                continue

            if op_type == "randombrightnesscontrast":
                ops.append(A.RandomBrightnessContrast(p=_prob(op, 0.5)))
                continue

            if op_type == "huesaturationvalue":
                ops.append(
                    A.HueSaturationValue(
                        hue_shift_limit=float(op.get("hue_shift_limit", 20)),
                        sat_shift_limit=float(op.get("sat_shift_limit", 30)),
                        val_shift_limit=float(op.get("val_shift_limit", 20)),
                        p=_prob(op, 0.5),
                    )
                )
                continue

            if op_type == "randomzoomout":
                max_scale = float(op.get("max_scale", 1.2))
                ops.append(A.RandomScale(scale_limit=(0.0, max_scale - 1.0), p=_prob(op, 0.5)))
                ops.append(A.PadIfNeeded(min_height=resize_h, min_width=resize_w, border_mode=cv2.BORDER_CONSTANT, p=1.0))
                continue

            if op_type == "randomioucrop":
                ops.append(A.RandomSizedBBoxSafeCrop(height=resize_h, width=resize_w, p=_prob(op, 0.8)))
                continue

            if op_type == "sanitizeboundingboxes":
                continue

            if op_type == "convertpilimage":
                continue

            if op_type == "convertboxes":
                continue

    if not ops:
        ops.append(A.Resize(height=default_size, width=default_size, p=1.0))

    return ConfigurableAlbumentations(ops=ops, use_masks=use_masks)
