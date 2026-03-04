from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch

from ...core import move_targets_to_device


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
            logger.warning("EMA copy skipped during validate due to state mismatch: {}", error)
            use_ema = False

    try:
        yield use_ema
    finally:
        if use_ema:
            ema_model.restore(unwrapped_model)


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
    running_box = 0.0
    running_cls = 0.0
    running_dfl = 0.0
    running_custom = 0.0
    component_sums: Dict[str, float] = {}
    num_steps = 0

    for images, targets in val_loader:
        images = images.to(accelerator.device, non_blocking=True)
        targets = move_targets_to_device(targets, accelerator.device)

        with accelerator.autocast():
            outputs = model(images, targets=targets)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())

        for key, value in loss_dict.items():
            if value is None:
                continue
            numeric = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            component_sums[str(key)] = component_sums.get(str(key), 0.0) + numeric

        parts = splitter.split(loss_dict)
        running_total += float(total_loss.detach().item())
        running_box += float(parts["box_loss"])
        running_cls += float(parts["cls_loss"])
        running_dfl += float(parts["dfl_loss"])
        running_custom += float(parts["custom_loss"])
        num_steps += 1

    denom = max(1, num_steps)
    metrics = {
        "loss": running_total / float(denom),
        "box_loss": running_box / float(denom),
        "cls_loss": running_cls / float(denom),
        "dfl_loss": running_dfl / float(denom),
        "custom_loss": running_custom / float(denom),
    }
    metrics.update({f"criterion/{key}": total / float(denom) for key, total in component_sums.items()})
    return metrics
