from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from infra.config import AppConfig
from infra.engine.evaluate import evaluate_predictions
from infra.vis import create_visualization_logger

from .eval_inference import run_eval_inference_loop
from .eval_model_metrics import generate_eval_model_metrics_bundle, log_eval_model_metrics_bundle
from .eval_reporting import populate_confusion_diagnostics, write_metrics_json as write_metrics_json_file
from .eval_sampling import build_eval_samples


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
) -> Dict[str, float]:
    output_root = Path(app_config.train.output_dir)
    inference_dir = output_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    eval_vis_dir = inference_dir / "eval"
    eval_vis_dir.mkdir(parents=True, exist_ok=True)

    vis_logger = create_visualization_logger(
        output_root=output_root,
        experiment_name=experiment_name,
        tensorboard_enabled=bool(app_config.runtime.visualization.tensorboard.enabled),
        tensorboard_log_dir=str(app_config.runtime.visualization.tensorboard.log_dir),
        tensorboard_host=str(app_config.runtime.visualization.tensorboard.host),
        tensorboard_port=int(app_config.runtime.visualization.tensorboard.port),
        tensorboard_start_service=bool(app_config.runtime.visualization.tensorboard.start_service),
        mlflow_enabled=bool(app_config.runtime.visualization.mlflow.enabled),
        mlflow_dir=str(app_config.runtime.visualization.mlflow.mlflow_dir),
        mlflow_tracking_backend=str(app_config.runtime.visualization.mlflow.tracking_backend),
        mlflow_sqlite_db_name=str(app_config.runtime.visualization.mlflow.sqlite_db_name),
        mlflow_host=str(app_config.runtime.visualization.mlflow.host),
        mlflow_port=int(app_config.runtime.visualization.mlflow.port),
        mlflow_start_service=bool(app_config.runtime.visualization.mlflow.start_service),
        execution_config=app_config.model_dump(mode="json"),
        logger_port=logger,
    )

    model_eval = model.deploy() if hasattr(model, "deploy") else model
    post_eval = postprocessor.deploy() if hasattr(postprocessor, "deploy") else postprocessor
    model_eval = model_eval.to(device).eval()

    samples = build_eval_samples(app_config.data.val_sets, app_config.data.mapping or {})
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

    if write_metrics_json:
        write_metrics_json_file(eval_vis_dir=eval_vis_dir, metrics=metrics, logger=logger)
        try:
            metrics_json_path = eval_vis_dir / "metrics.json"
            loaded_metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
            vis_logger.log_artifact(file_path=metrics_json_path, artifact_path="eval")
            vis_logger.log_text(
                tag="eval/metrics_json",
                text=json.dumps(loaded_metrics, indent=2, ensure_ascii=False),
                step=0,
            )
            logger.info("Logged metrics.json payload to visualization backends from {}", metrics_json_path)
        except Exception as error:
            logger.warning("Failed to log metrics.json payload to visualization backends: {}", error)

    vis_logger.log_metrics(metrics={f"eval/{key}": float(value) for key, value in metrics.items()}, step=0)

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
    )
    log_eval_model_metrics_bundle(
        output_root=output_root,
        vis_logger=vis_logger,
        step=0,
        metrics=metrics,
        confusion_scores=confusion_scores,
        confusion_labels=confusion_labels,
    )
    vis_logger.close()

    return metrics
