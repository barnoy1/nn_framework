from __future__ import annotations

from typing import Dict

import torch

from ..core import move_targets_to_device
from .training import compute_validation_loss_components, save_train_batch_visualization


def split_loss_components(trainer, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return trainer._loss_splitter.split(loss_dict)


@torch.no_grad()
def compute_validation_loss_components_for_trainer(trainer) -> Dict[str, float]:
    return compute_validation_loss_components(
        model=trainer.model,
        criterion=trainer.criterion,
        val_loader=trainer.val_loader,
        accelerator=trainer.accelerator,
        splitter=trainer._loss_splitter,
    )


def save_train_batch_visualization_for_trainer(trainer, images: torch.Tensor, targets, step: int) -> None:
    output_root = trainer.app_config.ensure_output_dir()
    save_train_batch_visualization(output_root=output_root, images=images, targets=targets, step=step)


def move_batch_to_device(trainer, images, targets):
    images = images.to(trainer.accelerator.device, non_blocking=True)
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
        return float(torch.cuda.memory_reserved(trainer.accelerator.device) / (1024 ** 3))
    return 0.0
