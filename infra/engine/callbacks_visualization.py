from __future__ import annotations

import json
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
        self._last_logged_artifacts: set[Path] = set()

    @staticmethod
    def _iter_files(path: Path) -> List[Path]:
        if not path.exists() or not path.is_dir():
            return []
        return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())

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

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _compute_detection_scores(matrix: np.ndarray) -> Dict[str, float]:
        if matrix is None or matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        class_count = int(matrix.shape[0] - 1)
        core = matrix[:class_count, :class_count]
        tp = float(np.trace(core))
        pred_non_bg = float(matrix[:, :class_count].sum())
        gt_non_bg = float(matrix[:class_count, :].sum())
        fp = max(0.0, pred_non_bg - tp)
        fn = max(0.0, gt_non_bg - tp)
        precision = ValidationVisualizationCallback._safe_ratio(tp, tp + fp)
        recall = ValidationVisualizationCallback._safe_ratio(tp, tp + fn)
        f1 = ValidationVisualizationCallback._safe_ratio(2.0 * precision * recall, precision + recall)
        accuracy = ValidationVisualizationCallback._safe_ratio(tp, tp + fp + fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def _log_output_artifacts(self) -> None:
        if self._logger is None:
            return

        root = self.output_dir
        artifact_candidates = [
            root / "results.csv",
            root / "results.png",
            root / "bbox_metrics.png",
            root / "BoxP_curve.png",
            root / "BoxR_curve.png",
            root / "BoxF1_curve.png",
            root / "BoxPR_curve.png",
            root / "confusion_matrix.png",
            root / "confusion_matrix_normalized.png",
        ]

        for artifact_path in artifact_candidates:
            resolved = artifact_path.resolve()
            if resolved.exists() and resolved not in self._last_logged_artifacts:
                self._logger.log_artifact(file_path=resolved, artifact_path="training")
                self._last_logged_artifacts.add(resolved)

    def _log_dataset_artifacts(self) -> None:
        if self._logger is None:
            return

        root = self.output_dir
        dataset_candidates = [
            root / "labels.png",
            root / "labels.jpg",
            root / "dataset" / "labels.png",
            root / "dataset" / "labels.jpg",
        ]
        dataset_candidates.extend(self._iter_files(root / "dataset"))

        for artifact_path in sorted({candidate.resolve() for candidate in dataset_candidates if candidate.exists()}):
            if artifact_path in self._last_logged_artifacts:
                continue
            self._logger.log_artifact(file_path=artifact_path, artifact_path="dataset")
            self._last_logged_artifacts.add(artifact_path)

    def _log_evaluation_artifacts(self, epoch: int) -> None:
        if self._logger is None:
            return

        epoch_prefix = f"evaluation/epoch_{epoch + 1:04d}"
        eval_dir = self.output_dir / "inference" / "eval"
        eval_files = self._iter_files(eval_dir)

        root_eval_candidates = [
            self.output_dir / "results.csv",
            self.output_dir / "results.png",
            self.output_dir / "bbox_metrics.png",
            self.output_dir / "BoxP_curve.png",
            self.output_dir / "BoxR_curve.png",
            self.output_dir / "BoxF1_curve.png",
            self.output_dir / "BoxPR_curve.png",
            self.output_dir / "confusion_matrix.png",
            self.output_dir / "confusion_matrix_normalized.png",
        ]
        eval_files.extend([candidate for candidate in root_eval_candidates if candidate.exists()])

        filtered_eval_files = [candidate for candidate in eval_files if candidate.name != "detections.json"]

        for artifact_path in sorted({candidate.resolve() for candidate in filtered_eval_files}):
            self._logger.log_artifact(file_path=artifact_path, artifact_path=epoch_prefix)

    @staticmethod
    def _build_validation_metric_payload(metrics: Dict[str, float]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        loss_keys = {"loss", "box_loss", "cls_loss", "dfl_loss", "custom_loss"}

        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue

            payload[f"val/{key}"] = numeric
            if key in loss_keys or key.startswith("criterion/"):
                payload[f"evaluation/losses/{key}"] = numeric
            else:
                payload[f"evaluation/coco/{key}"] = numeric

        return payload

    def on_train_start(self, trainer: "Trainer") -> None:
        if not trainer.accelerator.is_main_process:
            return
        vis_cfg = trainer.app_config.runtime.visualization
        output_root_resolved = self.output_dir.resolve()
        shared_tracking_dir = output_root_resolved.parent if "__" in output_root_resolved.name else output_root_resolved
        self._logger = create_visualization_logger(
            output_root=self.output_dir,
            experiment_name=self.experiment_name,
            tensorboard_enabled=self.tensorboard_enabled,
            tensorboard_log_dir=self.tensorboard_log_dir,
            tensorboard_host=str(vis_cfg.tensorboard.host),
            tensorboard_port=int(vis_cfg.tensorboard.port),
            tensorboard_start_service=bool(vis_cfg.tensorboard.start_service),
            mlflow_enabled=self.mlflow_enabled,
            mlflow_dir=str(shared_tracking_dir),
            mlflow_tracking_backend=str(vis_cfg.mlflow.tracking_backend),
            mlflow_sqlite_db_name=str(vis_cfg.mlflow.sqlite_db_name),
            mlflow_host=str(vis_cfg.mlflow.host),
            mlflow_port=int(vis_cfg.mlflow.port),
            mlflow_start_service=bool(vis_cfg.mlflow.start_service),
            execution_config=trainer.app_config.model_dump(mode="json"),
            logger_port=trainer.logger,
        )

    def on_batch_end(self, trainer: "Trainer", epoch: int, step: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process or self._logger is None:
            return
        self._logger.log_metrics(metrics=metrics, step=trainer.global_step)

    def on_validation_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process:
            return
        if self._logger is None:
            return

        validation_payload = self._build_validation_metric_payload(metrics)
        if validation_payload:
            self._logger.log_metrics(metrics=validation_payload, step=epoch + 1)

        matrix = getattr(trainer, "last_validation_confusion_matrix", None)
        labels = list(getattr(trainer, "last_validation_confusion_labels", []) or [])
        if matrix is not None and labels:
            scores = self._compute_detection_scores(matrix)
            self._logger.log_metrics(
                metrics={f"val/confusion/{key}": float(value) for key, value in scores.items()},
                step=epoch + 1,
            )
            self._logger.log_metrics(
                metrics={f"evaluation/{key}": float(value) for key, value in scores.items()},
                step=epoch + 1,
            )
            self._logger.log_text(
                tag="val/confusion_labels",
                text=json.dumps(labels, ensure_ascii=False),
                step=epoch + 1,
            )

        eval_dir = self.output_dir / "inference" / "eval"
        for artifact_name in ("metrics.json", "detections.json"):
            artifact_path = eval_dir / artifact_name
            resolved_artifact = artifact_path.resolve()
            if resolved_artifact.exists():
                self._logger.log_text(
                    tag=f"eval/{artifact_name}",
                    text=f"path={resolved_artifact}",
                    step=epoch + 1,
                )

        self._log_evaluation_artifacts(epoch)
        self._log_dataset_artifacts()
        self._log_output_artifacts()

        samples = list(getattr(trainer, "last_validation_visual_samples", []) or [])[: self.num_samples]
        if not samples:
            return

        panel = self._compose_grid(samples)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._save_dir / f"val_epoch_{epoch + 1:04d}.jpg"
        Image.fromarray(panel).save(output_path)

        self._logger.log_image(tag="train/val_visualization_grid", image=panel, step=epoch + 1)
        trainer.logger.info("Saved validation visualization grid ({} samples) to {}", len(samples), output_path)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]) -> None:
        if not trainer.accelerator.is_main_process or self._logger is None:
            return
        numeric_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric_metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if numeric_metrics:
            self._logger.log_metrics(metrics=numeric_metrics, step=epoch + 1)
        self._log_dataset_artifacts()
        self._log_output_artifacts()

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._logger is not None:
            self._logger.close()
            self._logger = None
