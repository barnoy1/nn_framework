from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
