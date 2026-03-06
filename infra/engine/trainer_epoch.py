from __future__ import annotations

from typing import Dict

import torch

from .trainer_display import create_epoch_progress_bar, log_yolo_header, yolo_progress_row
from .trainer_utils import (
    batch_instances,
    gpu_mem_reserved_gb,
    image_size_text,
    move_batch_to_device,
    save_train_batch_visualization_for_trainer,
    split_loss_components,
    targets_have_valid_boxes,
    warn_unmatched_configured_losses,
)


def train_one_epoch(trainer, epoch: int) -> Dict[str, float]:
    trainer.model.train()
    running_loss = 0.0
    running_box_loss = 0.0
    running_cls_loss = 0.0
    running_dfl_loss = 0.0
    running_custom_loss = 0.0
    running_instances = 0.0
    last_size = "-"
    component_sums: Dict[str, float] = {}
    num_steps = 0

    is_main_process = bool(trainer.accelerator.is_main_process)
    if is_main_process:
        log_yolo_header(trainer.logger)

    train_iterable = trainer.train_loader
    pbar = None
    if is_main_process:
        pbar = create_epoch_progress_bar(trainer.train_loader)
        train_iterable = pbar

    for step, (images, targets) in enumerate(train_iterable):
        images, targets = move_batch_to_device(trainer, images, targets)
        current_instances = batch_instances(targets)
        current_size = image_size_text(images)
        last_size = current_size

        if not targets_have_valid_boxes(targets):
            if trainer.accelerator.is_main_process:
                trainer.logger.warning("Skipping batch with invalid target boxes at epoch={} step={}", epoch, step)
            continue

        if trainer.model_wrapper is not None:
            trainer.model_wrapper.configure_fixed_dn_num_group(
                model=trainer.accelerator.unwrap_model(trainer.model),
                targets=targets,
                dn_num_group=trainer.app_config.model.dn_num_group,
            )

        try:
            with trainer.accelerator.autocast():
                outputs = trainer.model(images, targets=targets)
                loss_dict = trainer.criterion(outputs, targets)
                loss = sum(loss_dict.values())
        except AssertionError as error:
            if trainer.accelerator.is_main_process:
                trainer.logger.warning(
                    "Skipping unstable batch due to matcher assertion at epoch={} step={}: {}",
                    epoch,
                    step,
                    error,
                )
            trainer.optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(loss):
            if trainer.accelerator.is_main_process:
                trainer.logger.warning(
                    "Skipping non-finite loss at epoch={} step={} value={}",
                    epoch,
                    step,
                    float(loss.detach().item()) if torch.is_tensor(loss) else loss,
                )
            trainer.optimizer.zero_grad(set_to_none=True)
            continue

        warn_unmatched_configured_losses(trainer, loss_dict)

        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.accelerator.backward(loss)
        if trainer.app_config.train.grad_clip_norm > 0:
            trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.app_config.train.grad_clip_norm)
        trainer.optimizer.step()

        if trainer.accelerator.is_main_process and epoch == 0 and step < 3 and step not in trainer._saved_train_batch_steps:
            save_train_batch_visualization_for_trainer(trainer, images=images, targets=targets, step=step)
            trainer._saved_train_batch_steps.add(step)

        running_loss += float(loss.detach().item())
        parts = split_loss_components(trainer, loss_dict)
        running_box_loss += float(parts["box_loss"])
        running_cls_loss += float(parts["cls_loss"])
        running_dfl_loss += float(parts["dfl_loss"])
        running_custom_loss += float(parts["custom_loss"])
        running_instances += float(current_instances)
        num_steps += 1
        trainer.global_step += 1

        metrics = {
            "train/loss": float(loss.detach().item()),
            "train/box_loss": float(parts["box_loss"]),
            "train/cls_loss": float(parts["cls_loss"]),
            "train/dfl_loss": float(parts["dfl_loss"]),
            "train/custom_loss": float(parts["custom_loss"]),
        }
        for loss_key, loss_value in loss_dict.items():
            if loss_value is None:
                continue
            numeric = float(loss_value.detach().item()) if torch.is_tensor(loss_value) else float(loss_value)
            metrics[f"train/criterion/{loss_key}"] = numeric
            component_sums[str(loss_key)] = component_sums.get(str(loss_key), 0.0) + numeric
        trainer.callbacks.on_batch_end(trainer, epoch, step, metrics)

        if pbar is not None:
            pbar.set_description(
                yolo_progress_row(
                    epoch_index=epoch,
                    total_epochs=trainer.total_epochs,
                    gpu_mem_gb=gpu_mem_reserved_gb(trainer),
                    box_loss=float(parts["box_loss"]),
                    cls_loss=float(parts["cls_loss"]),
                    dfl_loss=float(parts["dfl_loss"]),
                    instances=int(current_instances),
                    image_size=current_size,
                ),
                refresh=False,
            )

        if step % trainer.app_config.train.log_every_n_steps == 0 and trainer.accelerator.is_main_process:
            trainer.logger.debug("epoch={} step={} loss={:.6f}", epoch, step, metrics["train/loss"])

    if pbar is not None:
        pbar.close()

    trainer.scheduler.step()
    denom = max(1, num_steps)
    avg_instances = running_instances / float(denom)
    epoch_metrics = {
        "loss": running_loss / float(denom),
        "box_loss": running_box_loss / float(denom),
        "cls_loss": running_cls_loss / float(denom),
        "dfl_loss": running_dfl_loss / float(denom),
        "custom_loss": running_custom_loss / float(denom),
    }
    epoch_metrics.update({f"criterion/{key}": total / float(denom) for key, total in component_sums.items()})

    if is_main_process:
        trainer.logger.info(
            "{}",
            yolo_progress_row(
                epoch_index=epoch,
                total_epochs=trainer.total_epochs,
                gpu_mem_gb=gpu_mem_reserved_gb(trainer),
                box_loss=float(epoch_metrics["box_loss"]),
                cls_loss=float(epoch_metrics["cls_loss"]),
                dfl_loss=float(epoch_metrics["dfl_loss"]),
                instances=int(round(avg_instances)),
                image_size=last_size,
            ),
        )

    return epoch_metrics
