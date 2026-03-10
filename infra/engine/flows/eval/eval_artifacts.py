from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from infra.config import AppConfig
from infra.engine.evaluate import evaluate_predictions
from infra.tracking import create_visualization_logger

from .eval_inference import run_eval_inference_loop
from .eval_model_metrics import (
    generate_eval_model_metrics_bundle,
    log_eval_model_metrics_bundle,
)
from .eval_reporting import populate_confusion_diagnostics
from .eval_sampling import build_eval_samples


EVAL_RESULTS_FIELDNAMES = [
    "epoch",
    "time",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/F1(B)",
    "metrics/accuracy(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/bbox_map_75",
    "val/bbox_mar_100",
]


def _prune_eval_non_image_files(eval_root_dir: Path) -> None:
    if not eval_root_dir.exists():
        return
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    for candidate in eval_root_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in image_suffixes:
            continue
        candidate.unlink(missing_ok=True)


def _build_eval_results_row(
    *, epoch: int, metrics: Dict[str, float], confusion_scores: Dict[str, float]
) -> Dict[str, float]:
    return {
        "epoch": float(epoch),
        "time": float(epoch),
        "metrics/precision(B)": float(confusion_scores.get("precision", 0.0)),
        "metrics/recall(B)": float(confusion_scores.get("recall", 0.0)),
        "metrics/F1(B)": float(confusion_scores.get("f1", 0.0)),
        "metrics/accuracy(B)": float(confusion_scores.get("accuracy", 0.0)),
        "metrics/mAP50(B)": float(metrics.get("bbox_map_50", 0.0)),
        "metrics/mAP50-95(B)": float(metrics.get("bbox_map", metrics.get("map", 0.0))),
        "val/bbox_map_75": float(metrics.get("bbox_map_75", 0.0)),
        "val/bbox_mar_100": float(metrics.get("bbox_mar_100", 0.0)),
    }


def _load_eval_results_history(results_csv: Path) -> Dict[int, Dict[str, float]]:
    rows_by_epoch: Dict[int, Dict[str, float]] = {}
    if not results_csv.exists():
        return rows_by_epoch

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            try:
                epoch = int(float(raw_row.get("epoch", 0.0)))
            except (TypeError, ValueError):
                continue
            row: Dict[str, float] = {}
            for field in EVAL_RESULTS_FIELDNAMES:
                try:
                    row[field] = float(raw_row.get(field, 0.0))
                except (TypeError, ValueError):
                    row[field] = 0.0
            rows_by_epoch[epoch] = row

    return rows_by_epoch


def _write_eval_results_history(
    results_csv: Path, rows_by_epoch: Dict[int, Dict[str, float]]
) -> None:
    ordered_epochs = sorted(rows_by_epoch.keys())
    with results_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_RESULTS_FIELDNAMES)
        writer.writeheader()
        for epoch in ordered_epochs:
            writer.writerow(rows_by_epoch[epoch])


def _update_eval_history_files(
    *,
    eval_root_dir: Path,
    epoch: int,
    metrics: Dict[str, float],
    confusion_scores: Dict[str, float],
) -> None:
    eval_root_dir.mkdir(parents=True, exist_ok=True)

    history_json = eval_root_dir / "metrics.json"
    history_entries: Dict[int, Dict[str, float]] = {}
    if history_json.exists():
        try:
            loaded = json.loads(history_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                loaded = [loaded]
            if isinstance(loaded, list):
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    try:
                        item_epoch = int(float(item.get("epoch", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    history_entries[item_epoch] = item
        except Exception:
            history_entries = {}

    entry = {"epoch": int(epoch)}
    entry.update({key: float(value) for key, value in metrics.items()})
    entry.update(
        {f"confusion_{key}": float(value) for key, value in confusion_scores.items()}
    )
    history_entries[int(epoch)] = entry

    ordered_entries = [
        history_entries[current_epoch]
        for current_epoch in sorted(history_entries.keys())
    ]
    history_json.write_text(
        json.dumps(ordered_entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results_csv = eval_root_dir / "results.csv"
    rows_by_epoch = _load_eval_results_history(results_csv)
    rows_by_epoch[int(epoch)] = _build_eval_results_row(
        epoch=int(epoch),
        metrics=metrics,
        confusion_scores=confusion_scores,
    )
    _write_eval_results_history(results_csv, rows_by_epoch)


def _build_eval_metric_payload(metrics: Dict[str, float]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    loss_keys = {"loss", "box_loss", "cls_loss", "dfl_loss", "custom_loss"}

    for key, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        payload[f"eval/{key}"] = numeric
        if key in loss_keys or key.startswith("criterion/"):
            payload[f"evaluation/losses/{key}"] = numeric
        else:
            payload[f"evaluation/coco/{key}"] = numeric

    return payload


def run_eval_artifacts(
    *,
    app_config: AppConfig,
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    logger,
    class_id_to_name: Dict[int, str],
    experiment_name: str,
    vis_samples: int = 16,
    score_thr: float = 0.3,
    image_epoch_suffix: Optional[int] = None,
    write_metrics_json: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
    use_deploy_model: bool = True,
) -> Dict[str, float]:
    output_root = Path(app_config.train.output_dir)
    output_root_resolved = output_root.resolve()
    shared_tracking_dir = (
        output_root_resolved.parent
        if "__" in output_root_resolved.name
        else output_root_resolved
    )
    inference_dir = output_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    eval_vis_dir = inference_dir / "eval"
    eval_vis_dir.mkdir(parents=True, exist_ok=True)
    _prune_eval_non_image_files(eval_vis_dir)
    log_step = int(image_epoch_suffix) if image_epoch_suffix is not None else 0

    vis_logger = create_visualization_logger(
        output_root=output_root,
        experiment_name=experiment_name,
        tensorboard_enabled=bool(app_config.runtime.visualization.tensorboard.enabled),
        tensorboard_log_dir=str(app_config.runtime.visualization.tensorboard.log_dir),
        tensorboard_host=str(app_config.runtime.visualization.tensorboard.host),
        tensorboard_port=int(app_config.runtime.visualization.tensorboard.port),
        tensorboard_start_service=bool(
            app_config.runtime.visualization.tensorboard.start_service
        ),
        mlflow_enabled=bool(app_config.runtime.visualization.mlflow.enabled),
        mlflow_dir=str(shared_tracking_dir),
        mlflow_tracking_backend=str(
            app_config.runtime.visualization.mlflow.tracking_backend
        ),
        mlflow_sqlite_db_name=str(
            app_config.runtime.visualization.mlflow.sqlite_db_name
        ),
        mlflow_host=str(app_config.runtime.visualization.mlflow.host),
        mlflow_port=int(app_config.runtime.visualization.mlflow.port),
        mlflow_start_service=bool(
            app_config.runtime.visualization.mlflow.start_service
        ),
        execution_config=app_config.model_dump(mode="json"),
        logger_port=logger,
    )

    model_eval = (
        model.deploy() if use_deploy_model and hasattr(model, "deploy") else model
    )
    post_eval = (
        postprocessor.deploy()
        if use_deploy_model and hasattr(postprocessor, "deploy")
        else postprocessor
    )
    model_eval = model_eval.to(device).eval()

    samples = build_eval_samples(
        app_config.data.val_sets, app_config.data.mapping or {}
    )
    eval_outputs = run_eval_inference_loop(
        app_config=app_config,
        samples=samples,
        model_eval=model_eval,
        post_eval=post_eval,
        device=device,
        class_id_to_name=class_id_to_name,
        score_thr=score_thr,
        vis_samples=vis_samples,
        image_epoch_suffix=image_epoch_suffix,
        eval_vis_dir=eval_vis_dir,
        vis_logger=vis_logger,
        logger=logger,
    )

    metrics = evaluate_predictions(
        predictions=eval_outputs["all_predictions"],
        targets=eval_outputs["all_targets_for_metric"],
        iou_types=app_config.data.iou_types,
    )

    diagnostics_payload: Dict[str, Any] = diagnostics if diagnostics is not None else {}

    populate_confusion_diagnostics(
        diagnostics=diagnostics_payload,
        confusion_events=eval_outputs["confusion_events"],
        class_id_to_name=class_id_to_name,
    )

    confusion_matrix = None
    confusion_labels = []
    confusion_matrix = diagnostics_payload.get("confusion_matrix")
    confusion_labels = list(diagnostics_payload.get("confusion_labels", []))

    output_root.mkdir(parents=True, exist_ok=True)
    confusion_scores = generate_eval_model_metrics_bundle(
        output_root=output_root,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        confusion_labels=confusion_labels,
        epoch=max(1, log_step),
    )
    _update_eval_history_files(
        eval_root_dir=output_root,
        epoch=max(1, log_step),
        metrics=metrics,
        confusion_scores=confusion_scores,
    )
    log_eval_model_metrics_bundle(
        output_root=output_root,
        vis_logger=vis_logger,
        step=log_step,
        metrics=metrics,
        confusion_scores=confusion_scores,
        confusion_labels=confusion_labels,
    )
    vis_logger.close()

    return metrics
