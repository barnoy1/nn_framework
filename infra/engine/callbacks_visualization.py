from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import numpy as np
from PIL import Image

from ..vis import create_visualization_logger
from .callbacks_base import Callback

if TYPE_CHECKING:
    from .trainer import Trainer


class ValidationVisualizationCallback(Callback):
    def __init__(
        self,
        output_dir: Path,
        num_samples: int,
        experiment_name: str,
        tensorboard_enabled: bool,
        tensorboard_log_dir: str,
        mlflow_enabled: bool,
    ) -> None:
        self.output_dir = output_dir
        self.num_samples = max(0, int(num_samples))
        self.experiment_name = experiment_name
        self.tensorboard_enabled = tensorboard_enabled
        self.tensorboard_log_dir = tensorboard_log_dir
        self.mlflow_enabled = mlflow_enabled
        self._logger = None
        self._save_dir = self.output_dir / "inference" / "validation"

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _compose_grid(images: List[np.ndarray]) -> np.ndarray:
        prepared = [ValidationVisualizationCallback._to_uint8_rgb(image) for image in images]
        max_height = max(image.shape[0] for image in prepared)
        max_width = max(image.shape[1] for image in prepared)

        count = len(prepared)
        cols = max(1, int(np.ceil(np.sqrt(count))))
        rows = int(np.ceil(count / cols))

        canvas = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)
        for index, image in enumerate(prepared):
            row = index // cols
            col = index % cols
            y0 = row * max_height
            x0 = col * max_width
            h, w = image.shape[:2]
            canvas[y0 : y0 + h, x0 : x0 + w] = image
        return canvas

    def on_train_start(self, trainer: "Trainer") -> None:
        if not trainer.accelerator.is_main_process or self.num_samples <= 0:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._logger = create_visualization_logger(
            output_root=self.output_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=self.tensorboard_enabled,
            tensorboard_log_dir=self.tensorboard_log_dir,
            mlflow_enabled=self.mlflow_enabled,
            mlflow_dir="mlflow",
            logger_port=trainer.logger,
        )

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process or self.num_samples <= 0:
            return
        if self._logger is None:
            return

        samples = list(getattr(trainer, "last_validation_visual_samples", []) or [])[: self.num_samples]
        if not samples:
            return

        panel = self._compose_grid(samples)
        output_path = self._save_dir / f"val_epoch_{epoch + 1:04d}.jpg"
        Image.fromarray(panel).save(output_path)

        self._logger.log_image(tag="train/val_visualization_grid", image=panel, step=epoch + 1)
        trainer.logger.info("Saved validation visualization grid ({} samples) to {}", len(samples), output_path)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None
