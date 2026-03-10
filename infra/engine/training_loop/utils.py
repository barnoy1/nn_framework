from __future__ import annotations

from typing import Dict

import torch

from ...core import move_targets_to_device
from infra.common.loss_aliases import canonical_loss_alias
from ..training import (
    align_images_to_model_input_channels,
    compute_validation_loss_components,
    save_eval_batch_visualization,
    save_train_batch_visualization,
    save_val_batch_visualization,
)


def split_loss_components(
    trainer, loss_dict: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    return trainer._loss_splitter.split(loss_dict)


def _loss_pattern_matches(loss_key: str, pattern: str) -> bool:
    lowered_key = str(loss_key).strip().lower()
    normalized = str(pattern).strip().lower()
    if not normalized:
        return False
    if normalized.endswith("_"):
        return lowered_key.startswith(normalized)
    return lowered_key == normalized or lowered_key.startswith(f"{normalized}_")


def warn_unmatched_configured_losses(
    trainer, loss_dict: Dict[str, torch.Tensor]
) -> None:
    if getattr(trainer, "_warned_unmatched_configured_losses", False):
        return

    configured_pairs = trainer.app_config.model.losses.criterion_pairs
    configured_specs = [
        *configured_pairs.iter_adapter_common(),
        *configured_pairs.iter_concrete_model(),
    ]

    active_patterns = [
        canonical_loss_alias(str(item.loss))
        for item in configured_specs
        if item.coef is None or float(item.coef) > 0.0
    ]
    if not active_patterns:
        trainer._warned_unmatched_configured_losses = True
        return

    produced_keys = [str(key).strip().lower() for key in loss_dict.keys()]
    unmatched = [
        pattern
        for pattern in active_patterns
        if not any(
            _loss_pattern_matches(loss_key, pattern) for loss_key in produced_keys
        )
    ]

    if unmatched and trainer.accelerator.is_main_process:
        trainer.logger.debug(
            "Configured losses not produced by model criterion (treated as zero in grouped metrics): {}",
            ", ".join(sorted(set(unmatched))),
        )

    trainer._warned_unmatched_configured_losses = True


@torch.no_grad()
def compute_validation_loss_components_for_trainer(trainer) -> Dict[str, float]:
    return compute_validation_loss_components(
        model=trainer.model,
        criterion=trainer.criterion,
        val_loader=trainer.val_loader,
        accelerator=trainer.accelerator,
        splitter=trainer._loss_splitter,
    )


def save_train_batch_visualization_for_trainer(
    trainer, images: torch.Tensor, targets, step: int
) -> None:
    output_root = trainer.app_config.ensure_output_dir()
    save_train_batch_visualization(
        output_root=output_root, images=images, targets=targets, step=step
    )


def save_eval_batch_visualizations_for_trainer(
    trainer,
    *,
    epoch_suffix: int | None = None,
    max_batches: int = 3,
) -> None:
    if not trainer.accelerator.is_main_process:
        return

    output_root = trainer.app_config.ensure_output_dir()
    num_samples = int(trainer.app_config.runtime.visualization.num_samples)
    saved = 0

    for step, (images, targets) in enumerate(trainer.val_loader):
        if step >= max_batches:
            break
        save_eval_batch_visualization(
            output_root=output_root,
            images=images,
            targets=targets,
            step=step,
            epoch_suffix=epoch_suffix,
            num_samples=num_samples,
        )
        saved += 1

    if saved > 0:
        trainer.logger.info(
            "Saved {} eval batch visualizations to {}", saved, output_root
        )


def save_val_batch_visualizations_for_trainer(
    trainer,
    *,
    epoch_suffix: int | None = None,
    max_batches: int = 3,
) -> None:
    if not trainer.accelerator.is_main_process:
        return

    output_root = trainer.app_config.ensure_output_dir()
    num_samples = int(trainer.app_config.runtime.visualization.num_samples)
    saved = 0

    for step, (images, targets) in enumerate(trainer.val_loader):
        if step >= max_batches:
            break
        save_val_batch_visualization(
            output_root=output_root,
            images=images,
            targets=targets,
            step=step,
            epoch_suffix=epoch_suffix,
            num_samples=num_samples,
        )
        saved += 1

    if saved > 0:
        trainer.logger.info(
            "Saved {} val batch visualizations to {}", saved, output_root
        )


def move_batch_to_device(trainer, images, targets):
    images = images.to(trainer.accelerator.device, non_blocking=True)
    images = align_images_to_model_input_channels(images=images, model=trainer.model)
    targets = move_targets_to_device(targets, trainer.accelerator.device)
    return images, targets


def targets_have_valid_boxes(targets) -> bool:
    for target in targets:
        boxes = target.get("boxes")
        if boxes is None or not torch.is_tensor(boxes):
            continue
        if boxes.numel() == 0:
            continue
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            return False
        if not torch.isfinite(boxes).all():
            return False
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        if (widths <= 0).any() or (heights <= 0).any():
            return False
    return True


def batch_instances(targets) -> int:
    instances = 0
    for target in targets:
        labels = target.get("labels")
        if labels is None:
            continue
        if torch.is_tensor(labels):
            instances += int(labels.numel())
        else:
            instances += len(labels)
    return instances


def image_size_text(images: torch.Tensor) -> str:
    if images.ndim >= 4:
        return f"{int(images.shape[-2])}x{int(images.shape[-1])}"
    return "-"


def gpu_mem_reserved_gb(trainer) -> float:
    if torch.cuda.is_available() and str(trainer.accelerator.device).startswith("cuda"):
        return float(torch.cuda.memory_reserved(trainer.accelerator.device) / (1024**3))
    return 0.0
