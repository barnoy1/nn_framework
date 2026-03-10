from __future__ import annotations

from typing import Dict, Optional

from ..flows.eval.eval_artifacts import run_eval_artifacts
from ..training import use_ema_weights_for_eval
from .utils import (
    compute_validation_loss_components_for_trainer,
    save_eval_batch_visualizations_for_trainer,
    save_val_batch_visualizations_for_trainer,
)


def validate_epoch(
    trainer, epoch: int, score_thr: Optional[float] = None
) -> Dict[str, float]:
    trainer.model.eval()
    resolved_score_thr = (
        float(score_thr)
        if score_thr is not None
        else float(trainer.app_config.runtime.common.score_threshold)
    )
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    with use_ema_weights_for_eval(trainer.ema_model, unwrapped_model, trainer.logger):
        save_val_batch_visualizations_for_trainer(
            trainer,
            epoch_suffix=None,
            max_batches=3,
        )
        class_id_to_name = {
            int(key): str(value)
            for key, value in (trainer.app_config.data.class_id_to_name or {}).items()
        }
        diagnostics: Dict[str, object] = {}
        metrics = run_eval_artifacts(
            app_config=trainer.app_config,
            model=unwrapped_model,
            postprocessor=trainer.postprocessor,
            device=trainer.accelerator.device,
            logger=trainer.logger,
            class_id_to_name=class_id_to_name,
            experiment_name=trainer.experiment_name,
            vis_samples=int(trainer.app_config.runtime.visualization.num_samples),
            score_thr=resolved_score_thr,
            image_epoch_suffix=epoch + 1,
            write_metrics_json=True,
            diagnostics=diagnostics,
            use_deploy_model=False,
        )
        val_loss_metrics = compute_validation_loss_components_for_trainer(trainer)
        metrics = metrics | val_loss_metrics
        trainer.last_validation_confusion_matrix = diagnostics.get("confusion_matrix")
        trainer.last_validation_confusion_labels = diagnostics.get(
            "confusion_labels", []
        )

    trainer.callbacks.on_validation_end(trainer, epoch, metrics)
    return metrics


def run_baseline_eval_sanity(
    trainer, epoch: int = -1, score_thr: Optional[float] = None
) -> Dict[str, float]:
    trainer.model.eval()
    resolved_score_thr = (
        float(score_thr)
        if score_thr is not None
        else float(trainer.app_config.runtime.common.score_threshold)
    )
    if trainer.accelerator.is_main_process:
        trainer.logger.info(
            "Pre-training evaluation procedure: running standalone-equivalent eval flow before optimizer updates "
            "(epoch_tag={}, score_threshold={:.3f})",
            epoch,
            resolved_score_thr,
        )
        save_eval_batch_visualizations_for_trainer(
            trainer,
            epoch_suffix=None,
            max_batches=3,
        )
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    class_id_to_name = {
        int(key): str(value)
        for key, value in (trainer.app_config.data.class_id_to_name or {}).items()
    }
    diagnostics: Dict[str, object] = {}
    metrics = run_eval_artifacts(
        app_config=trainer.app_config,
        model=unwrapped_model,
        postprocessor=trainer.postprocessor,
        device=trainer.accelerator.device,
        logger=trainer.logger,
        class_id_to_name=class_id_to_name,
        experiment_name=trainer.experiment_name,
        vis_samples=int(trainer.app_config.runtime.visualization.num_samples),
        score_thr=resolved_score_thr,
        image_epoch_suffix=epoch + 1,
        write_metrics_json=True,
        diagnostics=diagnostics,
        use_deploy_model=False,
    )
    trainer.last_validation_confusion_matrix = diagnostics.get("confusion_matrix")
    trainer.last_validation_confusion_labels = diagnostics.get("confusion_labels", [])
    trainer.callbacks.on_validation_end(trainer, epoch, metrics)
    return metrics
