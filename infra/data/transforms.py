from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
import math

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


class ApplyColorOnly:
    def __init__(self, augment: A.BasicTransform):
        self.augment = augment

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        channels = 1 if image.ndim == 2 else int(image.shape[2]) if image.ndim >= 3 else 1
        if channels < 3:
            return image
        return self.augment(image=image)["image"]


class ApplyToGrayIfNeeded:
    def __init__(self, num_output_channels: int):
        self.num_output_channels = int(num_output_channels)
        self._to_gray = A.ToGray(num_output_channels=1, p=1.0)

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        channels = 1 if image.ndim == 2 else int(image.shape[2]) if image.ndim >= 3 else 1

        if channels >= 3:
            gray = self._to_gray(image=image)["image"]
        elif image.ndim == 2:
            gray = image
        else:
            gray = image[:, :, 0]

        if self.num_output_channels == 1:
            return gray if gray.ndim == 2 else gray[:, :, 0]

        if self.num_output_channels == 3:
            if gray.ndim == 2:
                return np.repeat(gray[:, :, None], 3, axis=2)
            return np.repeat(gray[:, :, :1], 3, axis=2)

        return gray


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
                jitter = A.ColorJitter(p=_prob(op, 0.5))
                ops.append(A.Lambda(image=ApplyColorOnly(jitter), p=1.0))
                continue

            if op_type == "randombrightnesscontrast":
                ops.append(A.RandomBrightnessContrast(p=_prob(op, 0.5)))
                continue

            if op_type == "clahe":
                clip_limit = op.get("clip_limit", 4.0)
                grid = op.get("tile_grid_size", [8, 8])
                if isinstance(grid, list) and len(grid) >= 2:
                    tile_grid_size = (int(grid[0]), int(grid[1]))
                else:
                    tile_grid_size = (8, 8)
                ops.append(
                    A.CLAHE(
                        clip_limit=float(clip_limit),
                        tile_grid_size=tile_grid_size,
                        p=_prob(op, 0.5),
                    )
                )
                continue

            if op_type == "randomgamma":
                gamma_limit = op.get("gamma_limit", [80, 120])
                if isinstance(gamma_limit, list) and len(gamma_limit) >= 2:
                    gamma_range = (float(gamma_limit[0]), float(gamma_limit[1]))
                else:
                    bound = float(gamma_limit)
                    gamma_range = (max(1.0, 100.0 - bound), 100.0 + bound)
                ops.append(A.RandomGamma(gamma_limit=gamma_range, p=_prob(op, 0.5)))
                continue

            if op_type == "gaussnoise":
                std_range_cfg = op.get("std_range")
                if isinstance(std_range_cfg, list) and len(std_range_cfg) >= 2:
                    std_range = (float(std_range_cfg[0]), float(std_range_cfg[1]))
                else:
                    var_limit = op.get("var_limit", [10.0, 50.0])
                    if isinstance(var_limit, list) and len(var_limit) >= 2:
                        var_min = float(var_limit[0])
                        var_max = float(var_limit[1])
                    else:
                        bound = float(var_limit)
                        var_min, var_max = (0.0, max(1.0, bound))
                    std_min = math.sqrt(max(0.0, var_min)) / 255.0
                    std_max = math.sqrt(max(0.0, var_max)) / 255.0
                    std_range = (std_min, std_max)

                mean_range_cfg = op.get("mean_range", [0.0, 0.0])
                if isinstance(mean_range_cfg, list) and len(mean_range_cfg) >= 2:
                    mean_range = (float(mean_range_cfg[0]), float(mean_range_cfg[1]))
                else:
                    mean_value = float(mean_range_cfg)
                    mean_range = (mean_value, mean_value)

                per_channel = bool(op.get("per_channel", True))
                noise_scale_factor = float(op.get("noise_scale_factor", 1.0))
                ops.append(
                    A.GaussNoise(
                        std_range=std_range,
                        mean_range=mean_range,
                        per_channel=per_channel,
                        noise_scale_factor=noise_scale_factor,
                        p=_prob(op, 0.3),
                    )
                )
                continue

            if op_type == "gaussianblur":
                blur_limit = op.get("blur_limit", [3, 5])
                if isinstance(blur_limit, list) and len(blur_limit) >= 2:
                    blur_range = (int(blur_limit[0]), int(blur_limit[1]))
                else:
                    limit = int(blur_limit)
                    blur_range = (max(1, limit), max(1, limit))
                ops.append(A.GaussianBlur(blur_limit=blur_range, p=_prob(op, 0.2)))
                continue

            if op_type == "huesaturationvalue":
                hsv = A.HueSaturationValue(
                    hue_shift_limit=float(op.get("hue_shift_limit", 20)),
                    sat_shift_limit=float(op.get("sat_shift_limit", 30)),
                    val_shift_limit=float(op.get("val_shift_limit", 20)),
                    p=_prob(op, 0.5),
                )
                ops.append(A.Lambda(image=ApplyColorOnly(hsv), p=1.0))
                continue

            if op_type == "togray":
                num_output_channels = int(op.get("num_output_channels", 3))
                ops.append(A.Lambda(image=ApplyToGrayIfNeeded(num_output_channels), p=_prob(op, 0.5)))
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
