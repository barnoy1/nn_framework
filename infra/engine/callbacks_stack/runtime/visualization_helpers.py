from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def iter_files(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def compose_grid(images: List[np.ndarray]) -> np.ndarray:
    prepared = [to_uint8_rgb(image) for image in images]
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


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_detection_scores(matrix: np.ndarray) -> Dict[str, float]:
    if matrix is None or matrix.size == 0 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    class_count = int(matrix.shape[0] - 1)
    core = matrix[:class_count, :class_count]
    tp = float(np.trace(core))
    pred_non_bg = float(matrix[:, :class_count].sum())
    gt_non_bg = float(matrix[:class_count, :].sum())
    fp = max(0.0, pred_non_bg - tp)
    fn = max(0.0, gt_non_bg - tp)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = safe_ratio(tp, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def build_validation_metric_payload(metrics: Dict[str, float]) -> Dict[str, float]:
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


def log_output_artifacts(*, output_dir: Path, logger, logged_artifacts: set[Path]) -> None:
    artifact_candidates = [
        output_dir / "results.csv",
        output_dir / "results.png",
        output_dir / "bbox_metrics.png",
        output_dir / "BoxP_curve.png",
        output_dir / "BoxR_curve.png",
        output_dir / "BoxF1_curve.png",
        output_dir / "BoxPR_curve.png",
        output_dir / "confusion_matrix.png",
        output_dir / "confusion_matrix_normalized.png",
    ]
    for artifact_path in artifact_candidates:
        resolved = artifact_path.resolve()
        if resolved.exists() and resolved not in logged_artifacts:
            logger.log_artifact(file_path=resolved, artifact_path="training")
            logged_artifacts.add(resolved)


def log_dataset_artifacts(*, output_dir: Path, logger, logged_artifacts: set[Path]) -> None:
    dataset_candidates = [
        output_dir / "labels.png",
        output_dir / "labels.jpg",
        output_dir / "dataset" / "labels.png",
        output_dir / "dataset" / "labels.jpg",
    ]
    dataset_candidates.extend(iter_files(output_dir / "dataset"))
    resolved_paths = {candidate.resolve() for candidate in dataset_candidates if candidate.exists()}

    for artifact_path in sorted(resolved_paths):
        if artifact_path in logged_artifacts:
            continue
        logger.log_artifact(file_path=artifact_path, artifact_path="dataset")
        logged_artifacts.add(artifact_path)


def log_batch_artifacts(*, output_dir: Path, logger, logged_artifacts: set[Path]) -> None:
    batch_roots = [
        (output_dir, "training/train_batches", "train_batch*.jpg"),
        (output_dir, "training/validation_batches", "val_batch*.jpg"),
        (output_dir, "training/eval_batches", "eval_batch*.jpg"),
        (output_dir / "inference" / "validation", "training/validation_grids", "val_epoch_*.jpg"),
    ]

    for root_dir, artifact_path, pattern in batch_roots:
        if not root_dir.exists():
            continue
        for candidate in sorted(root_dir.glob(pattern)):
            resolved = candidate.resolve()
            if resolved in logged_artifacts:
                continue
            logger.log_artifact(file_path=resolved, artifact_path=artifact_path)
            logged_artifacts.add(resolved)


def log_evaluation_artifacts(*, output_dir: Path, epoch: int, logger, logged_artifacts: set[Path]) -> None:
    epoch_prefix = f"evaluation/epoch_{epoch + 1:04d}"
    eval_dir = output_dir / "inference" / "eval"
    epoch_token = f"__epoch_{epoch + 1:04d}.png"

    epoch_candidates = [candidate for candidate in iter_files(eval_dir) if candidate.name.endswith(epoch_token)]
    for artifact_path in epoch_candidates:
        resolved = artifact_path.resolve()
        if resolved in logged_artifacts:
            continue
        logger.log_artifact(file_path=resolved, artifact_path=epoch_prefix)
        logged_artifacts.add(resolved)

    best_dir = output_dir / "best"
    for artifact_path in iter_files(best_dir):
        resolved = artifact_path.resolve()
        if resolved in logged_artifacts:
            continue
        logger.log_artifact(file_path=resolved, artifact_path="evaluation/best")
        logged_artifacts.add(resolved)


def log_accumulated_eval_history_artifacts(*, output_dir: Path, logger) -> None:
    history_candidates = [
        output_dir / "metrics.json",
        output_dir / "results.csv",
        output_dir / "results.png",
        output_dir / "bbox_metrics.png",
        output_dir / "BoxP_curve.png",
        output_dir / "BoxR_curve.png",
        output_dir / "BoxF1_curve.png",
        output_dir / "BoxPR_curve.png",
        output_dir / "confusion_matrix.png",
        output_dir / "confusion_matrix_normalized.png",
        output_dir / "best_epoch.json",
    ]
    for artifact_path in history_candidates:
        resolved = artifact_path.resolve()
        if resolved.exists() and resolved.is_file():
            logger.log_artifact(file_path=resolved, artifact_path="evaluation/history")


def sync_execution_tree_artifacts(*, output_dir: Path, logger, artifact_mtimes: dict[Path, int]) -> None:
    if not output_dir.exists():
        return
    for candidate in sorted(output_dir.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            mtime_ns = candidate.stat().st_mtime_ns
        except OSError:
            continue
        resolved = candidate.resolve()
        previous_mtime = artifact_mtimes.get(resolved)
        if previous_mtime is not None and previous_mtime == mtime_ns:
            continue

        relative_path = resolved.relative_to(output_dir.resolve())
        parent = relative_path.parent.as_posix()
        artifact_path = "" if parent == "." else parent
        logger.log_artifact(file_path=resolved, artifact_path=artifact_path)
        artifact_mtimes[resolved] = mtime_ns
