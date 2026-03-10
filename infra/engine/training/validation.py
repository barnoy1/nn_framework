from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch

from ...core import move_targets_to_device
from .input_channels import align_images_to_model_input_channels


@contextmanager
def use_ema_weights_for_eval(ema_model, unwrapped_model, logger):
    use_ema = ema_model is not None
    if use_ema:
        try:
            ema_model.store(unwrapped_model)
            ema_model.copy_to(unwrapped_model)
        except RuntimeError as error:
            try:
                ema_model.restore(unwrapped_model)
            except Exception:
                pass
            logger.warning(
                "EMA copy skipped during validate due to state mismatch: {}", error
            )
            use_ema = False

    try:
        yield use_ema
    finally:
        if use_ema:
            try:
                ema_model.restore(unwrapped_model)
            except RuntimeError as error:
                logger.warning(
                    "EMA restore skipped after validate due to state mismatch: {}",
                    error,
                )


@torch.no_grad()
def compute_validation_loss_components(
    *,
    model,
    criterion,
    val_loader,
    accelerator,
    splitter,
) -> Dict[str, float]:
    model.eval()
    running_total = 0.0
    running_parts: Dict[str, float] = {}
    component_sums: Dict[str, float] = {}
    num_steps = 0

    for images, targets in val_loader:
        images = images.to(accelerator.device, non_blocking=True)
        images = align_images_to_model_input_channels(images=images, model=model)
        targets = move_targets_to_device(targets, accelerator.device)

        with accelerator.autocast():
            outputs = model(images, targets=targets)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())

        for key, value in loss_dict.items():
            if value is None:
                continue
            numeric = (
                float(value.detach().item()) if torch.is_tensor(value) else float(value)
            )
            component_sums[str(key)] = component_sums.get(str(key), 0.0) + numeric

        parts = splitter.split(loss_dict)
        running_total += float(total_loss.detach().item())
        for key, value in parts.items():
            running_parts[key] = running_parts.get(key, 0.0) + float(value)
        num_steps += 1

    denom = max(1, num_steps)
    metrics = {"loss": running_total / float(denom)}
    metrics.update({key: total / float(denom) for key, total in running_parts.items()})
    metrics.setdefault("custom_loss", 0.0)
    metrics.update(
        {
            f"criterion/{key}": total / float(denom)
            for key, total in component_sums.items()
        }
    )
    return metrics
