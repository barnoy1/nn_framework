from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np

from ..config import AugConfig


@dataclass
class TransformResult:
    image: np.ndarray
    bboxes: List[List[float]]
    class_labels: List[int]
    masks: Optional[List[np.ndarray]] = None


class DynamicAlbumentations:
    def __init__(self, config: AugConfig, use_masks: bool = False):
        self.config = config
        self.use_masks = use_masks
        self.current_stage = "heavy"
        self._heavy = self._build_heavy_pipeline()
        self._light = self._build_light_pipeline()

    def _build_common(self) -> List[Any]:
        return [
            A.HorizontalFlip(p=self.config.horizontal_flip_prob),
            A.RandomBrightnessContrast(p=self.config.color_jitter_prob),
        ]

    def _build_heavy_pipeline(self) -> A.Compose:
        ops = [
            A.LongestMaxSize(max_size=self.config.image_size),
            A.PadIfNeeded(min_height=self.config.image_size, min_width=self.config.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=(self.config.heavy_scale_min - 1.0, self.config.heavy_scale_max - 1.0), p=0.9),
            A.RandomSizedBBoxSafeCrop(height=self.config.image_size, width=self.config.image_size, p=0.8),
            *self._build_common(),
        ]
        return self._compose(ops)

    def _build_light_pipeline(self) -> A.Compose:
        ops = [
            A.LongestMaxSize(max_size=self.config.image_size),
            A.PadIfNeeded(min_height=self.config.image_size, min_width=self.config.image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomScale(scale_limit=(self.config.light_scale_min - 1.0, self.config.light_scale_max - 1.0), p=0.4),
            A.CenterCrop(height=self.config.image_size, width=self.config.image_size, p=0.7),
            *self._build_common(),
        ]
        return self._compose(ops)

    def _compose(self, ops: List[Any]) -> A.Compose:
        return A.Compose(
            ops,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.05),
        )

    def update_augmentation(self, epoch: int, total_epochs: int) -> None:
        switch_epoch = int(total_epochs * self.config.switch_epoch_ratio)
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
