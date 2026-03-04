from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from ..adapters import LoguruLoggerAdapter
from ..core import move_targets_to_device
from .callbacks import CallbackList
from ..config import AppConfig
from .flows.eval.eval_artifacts import run_eval_artifacts
from .model import ModelWrapperAdapter
from .training import (
    LossComponentSplitter,
    compute_validation_loss_components,
    save_train_batch_visualization,
    use_ema_weights_for_eval,
)


class Trainer:
    def __init__(
        self,
        app_config: AppConfig,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        postprocessor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader,
        val_loader,
        callbacks: CallbackList,
        ema_model=None,
        model_wrapper: Optional[ModelWrapperAdapter] = None,
        logger_port=None,
        experiment_name: str = "experiment",
    ) -> None:
        self.app_config = app_config
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks
        self.ema_model = ema_model
        self.model_wrapper = model_wrapper
        self.experiment_name = experiment_name
        self.logger = logger_port or LoguruLoggerAdapter()

        set_seed(self.app_config.train.seed)

        self.accelerator = Accelerator(mixed_precision=self.app_config.train.mixed_precision)
        if self.ema_model is not None:
            self.accelerator.register_for_checkpointing(self.ema_model)

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.ema_model.align_to_model(unwrapped_model)
            self.ema_model.copy_from(unwrapped_model)

        self.global_step = 0
        self.current_epoch = 0
        self._saved_train_batch_steps: set[int] = set()
        self.last_validation_confusion_matrix: Optional[np.ndarray] = None
        self.last_validation_confusion_labels: list[str] = []
        self.total_epochs = (
            int(self.app_config.runtime.epoches)
            if self.app_config.runtime.epoches is not None
            else int(self.app_config.train.epochs)
        )
        self._loss_splitter = LossComponentSplitter.from_config(self.app_config)

    def _split_loss_components(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._loss_splitter.split(loss_dict)

    @torch.no_grad()
    def _compute_validation_loss_components(self) -> Dict[str, float]:
        return compute_validation_loss_components(
            model=self.model,
            criterion=self.criterion,
            val_loader=self.val_loader,
            accelerator=self.accelerator,
            splitter=self._loss_splitter,
        )

    def _save_train_batch_visualization(self, images: torch.Tensor, targets, step: int) -> None:
        output_root = self.app_config.ensure_output_dir()
        save_train_batch_visualization(output_root=output_root, images=images, targets=targets, step=step)

    @staticmethod
    def _targets_have_valid_boxes(targets) -> bool:
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

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_box_loss = 0.0
        running_cls_loss = 0.0
        running_dfl_loss = 0.0
        running_custom_loss = 0.0
        component_sums: Dict[str, float] = {}
        num_steps = 0

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.accelerator.device, non_blocking=True)
            targets = move_targets_to_device(targets, self.accelerator.device)

            if not self._targets_have_valid_boxes(targets):
                if self.accelerator.is_main_process:
                    self.logger.warning("Skipping batch with invalid target boxes at epoch={} step={}", epoch, step)
                continue

            if self.model_wrapper is not None:
                self.model_wrapper.configure_fixed_dn_num_group(
                    model=self.accelerator.unwrap_model(self.model),
                    targets=targets,
                    dn_num_group=self.app_config.model.dn_num_group,
                )

            try:
                with self.accelerator.autocast():
                    outputs = self.model(images, targets=targets)
                    loss_dict = self.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
            except AssertionError as error:
                if self.accelerator.is_main_process:
                    self.logger.warning(
                        "Skipping unstable batch due to matcher assertion at epoch={} step={}: {}",
                        epoch,
                        step,
                        error,
                    )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if not torch.isfinite(loss):
                if self.accelerator.is_main_process:
                    self.logger.warning(
                        "Skipping non-finite loss at epoch={} step={} value={}",
                        epoch,
                        step,
                        float(loss.detach().item()) if torch.is_tensor(loss) else loss,
                    )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            if self.app_config.train.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.app_config.train.grad_clip_norm)
            self.optimizer.step()

            if self.accelerator.is_main_process and epoch == 0 and step < 3 and step not in self._saved_train_batch_steps:
                self._save_train_batch_visualization(images=images, targets=targets, step=step)
                self._saved_train_batch_steps.add(step)

            running_loss += float(loss.detach().item())
            parts = self._split_loss_components(loss_dict)
            running_box_loss += float(parts["box_loss"])
            running_cls_loss += float(parts["cls_loss"])
            running_dfl_loss += float(parts["dfl_loss"])
            running_custom_loss += float(parts["custom_loss"])
            num_steps += 1
            self.global_step += 1

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
            self.callbacks.on_batch_end(self, epoch, step, metrics)

            if step % self.app_config.train.log_every_n_steps == 0 and self.accelerator.is_main_process:
                self.logger.info("epoch={} step={} loss={:.6f}", epoch, step, metrics["train/loss"])

        self.scheduler.step()
        denom = max(1, num_steps)
        epoch_metrics = {
            "loss": running_loss / float(denom),
            "box_loss": running_box_loss / float(denom),
            "cls_loss": running_cls_loss / float(denom),
            "dfl_loss": running_dfl_loss / float(denom),
            "custom_loss": running_custom_loss / float(denom),
        }
        epoch_metrics.update({f"criterion/{key}": total / float(denom) for key, total in component_sums.items()})
        return epoch_metrics

    @torch.no_grad()
    def validate(self, epoch: int, score_thr: Optional[float] = None) -> Dict[str, float]:
        self.model.eval()
        resolved_score_thr = (
            float(score_thr)
            if score_thr is not None
            else float(self.app_config.runtime.common.score_threshold)
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with use_ema_weights_for_eval(self.ema_model, unwrapped_model, self.logger):
            class_id_to_name = {int(key): str(value) for key, value in (self.app_config.data.class_id_to_name or {}).items()}
            diagnostics: Dict[str, object] = {}
            metrics = run_eval_artifacts(
                app_config=self.app_config,
                model=unwrapped_model,
                postprocessor=self.postprocessor,
                device=self.accelerator.device,
                logger=self.logger,
                class_id_to_name=class_id_to_name,
                experiment_name=self.experiment_name,
                vis_samples=int(self.app_config.runtime.visualization.num_samples),
                score_thr=resolved_score_thr,
                image_epoch_suffix=epoch + 1,
                write_metrics_json=True,
                diagnostics=diagnostics,
            )
            val_loss_metrics = self._compute_validation_loss_components()
            metrics = metrics | val_loss_metrics
            self.last_validation_confusion_matrix = diagnostics.get("confusion_matrix")
            self.last_validation_confusion_labels = diagnostics.get("confusion_labels", [])

        self.callbacks.on_validation_end(self, epoch, metrics)
        return metrics

    @torch.no_grad()
    def run_baseline_eval_sanity(self, epoch: int = -1, score_thr: Optional[float] = None) -> Dict[str, float]:
        self.model.eval()
        resolved_score_thr = (
            float(score_thr)
            if score_thr is not None
            else float(self.app_config.runtime.common.score_threshold)
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Pre-training evaluation procedure: running standalone-equivalent eval flow before optimizer updates "
                "(epoch_tag={}, score_threshold={:.3f})",
                epoch,
                resolved_score_thr,
            )
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        class_id_to_name = {int(key): str(value) for key, value in (self.app_config.data.class_id_to_name or {}).items()}
        diagnostics: Dict[str, object] = {}
        metrics = run_eval_artifacts(
            app_config=self.app_config,
            model=unwrapped_model,
            postprocessor=self.postprocessor,
            device=self.accelerator.device,
            logger=self.logger,
            class_id_to_name=class_id_to_name,
            experiment_name=self.experiment_name,
            vis_samples=int(self.app_config.runtime.visualization.num_samples),
            score_thr=resolved_score_thr,
            image_epoch_suffix=epoch + 1,
            write_metrics_json=True,
            diagnostics=diagnostics,
        )
        self.last_validation_confusion_matrix = diagnostics.get("confusion_matrix")
        self.last_validation_confusion_labels = diagnostics.get("confusion_labels", [])
        self.callbacks.on_validation_end(self, epoch, metrics)
        return metrics

    def fit(self) -> None:
        self.callbacks.on_train_start(self)
        try:
            if self.accelerator.is_main_process:
                self.logger.info("Running baseline evaluation sanity-check before optimizer updates")
            baseline_metrics = self.run_baseline_eval_sanity(epoch=-1)
            if self.accelerator.is_main_process:
                self.logger.info("baseline metrics={}", {f"val_{k}": v for k, v in baseline_metrics.items()})

            first_epoch_in_run = True
            for epoch in range(self.total_epochs):
                self.current_epoch = epoch
                self.callbacks.on_epoch_start(self, epoch)

                train_metrics = self._train_one_epoch(epoch)

                val_metrics: Dict[str, float] = {}
                should_validate = first_epoch_in_run or ((epoch + 1) % self.app_config.train.val_every_n_epochs == 0)
                if should_validate:
                    val_metrics = self.validate(epoch)
                first_epoch_in_run = False

                merged_metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {
                    f"val_{k}": v for k, v in val_metrics.items()
                }
                self.callbacks.on_epoch_end(self, epoch, merged_metrics)

                if self.accelerator.is_main_process:
                    self.logger.info("epoch={} metrics={}", epoch, merged_metrics)
        finally:
            self.callbacks.on_train_end(self)
