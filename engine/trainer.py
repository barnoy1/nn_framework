from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from .callbacks import CallbackList
from ..config import AppConfig
from .evaluate import evaluate_predictions
from ..model import ModelWrapperAdapter
from ..utils.log import logger


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
            self.ema_model.align_to_model(self.accelerator.unwrap_model(self.model))

        self.global_step = 0
        self.current_epoch = 0

    def _move_targets_to_device(self, targets: List[Dict]) -> List[Dict]:
        device_targets: List[Dict] = []
        for target in targets:
            packed = {}
            for key, value in target.items():
                if isinstance(value, torch.Tensor):
                    packed[key] = value.to(self.accelerator.device, non_blocking=True)
                else:
                    packed[key] = value
            device_targets.append(packed)
        return device_targets

    @staticmethod
    def _to_result_list(outputs, postprocessor, orig_sizes):
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            return outputs

        if isinstance(outputs, dict):
            processed = postprocessor(outputs, orig_sizes)
        elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
            labels, boxes, scores = outputs
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
                boxes = boxes.unsqueeze(0)
                scores = scores.unsqueeze(0)
            return [
                {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
                for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
            ]
        else:
            processed = postprocessor(outputs, orig_sizes)

        if isinstance(processed, list):
            return processed

        if isinstance(processed, (tuple, list)) and len(processed) == 3:
            labels, boxes, scores = processed
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
                boxes = boxes.unsqueeze(0)
                scores = scores.unsqueeze(0)
            return [
                {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
                for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
            ]

        if isinstance(processed, dict) and {"labels", "boxes", "scores"}.issubset(set(processed.keys())):
            labels = processed["labels"]
            boxes = processed["boxes"]
            scores = processed["scores"]
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
                boxes = boxes.unsqueeze(0)
                scores = scores.unsqueeze(0)
            return [
                {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
                for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
            ]

        raise TypeError(f"Unsupported validation output format: {type(processed)}")

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        num_steps = 0

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.accelerator.device, non_blocking=True)
            targets = self._move_targets_to_device(targets)

            if self.model_wrapper is not None:
                self.model_wrapper.configure_fixed_dn_num_group(
                    model=self.accelerator.unwrap_model(self.model),
                    targets=targets,
                    dn_num_group=self.app_config.model.dn_num_group,
                )

            with self.accelerator.autocast():
                outputs = self.model(images, targets=targets)
                loss_dict = self.criterion(outputs, targets)
                loss = sum(loss_dict.values())

            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            if self.app_config.train.grad_clip_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.app_config.train.grad_clip_norm)
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            num_steps += 1
            self.global_step += 1

            metrics = {"train/loss": float(loss.detach().item())}
            self.callbacks.on_batch_end(self, epoch, step, metrics)

            if step % self.app_config.train.log_every_n_steps == 0 and self.accelerator.is_main_process:
                logger.info("epoch={} step={} loss={:.6f}", epoch, step, metrics["train/loss"])

        self.scheduler.step()
        return {"loss": running_loss / max(1, num_steps)}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        use_ema = self.ema_model is not None
        if use_ema:
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                self.ema_model.store(unwrapped)
                self.ema_model.copy_to(unwrapped)
            except RuntimeError as error:
                logger.warning("EMA copy skipped during validate due to state mismatch: {}", error)
                use_ema = False

        all_predictions: List[Dict[str, torch.Tensor]] = []
        all_targets_for_metric: List[Dict[str, torch.Tensor]] = []

        for images, targets in self.val_loader:
            images = images.to(self.accelerator.device, non_blocking=True)
            targets = self._move_targets_to_device(targets)

            outputs = self.model(images)
            orig_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
            results = self._to_result_list(outputs, self.postprocessor, orig_sizes)

            for prediction, target in zip(results, targets):
                pred = {
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                    "labels": prediction["labels"].detach().cpu().long(),
                }
                if "masks" in prediction:
                    pred["masks"] = prediction["masks"].detach().cpu().bool()
                all_predictions.append(pred)

                gt = {
                    "boxes": target["boxes_xyxy"].detach().cpu(),
                    "labels": target["labels"].detach().cpu().long(),
                }
                if "masks" in target:
                    gt["masks"] = target["masks"].detach().cpu().bool()
                all_targets_for_metric.append(gt)

        metrics = evaluate_predictions(
            predictions=all_predictions,
            targets=all_targets_for_metric,
            iou_types=self.app_config.data.iou_types,
        )

        if use_ema:
            self.ema_model.restore(self.accelerator.unwrap_model(self.model))

        self.callbacks.on_validation_end(self, epoch, metrics)
        return metrics

    def fit(self) -> None:
        self.callbacks.on_train_start(self)

        for epoch in range(self.app_config.train.epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_start(self, epoch)

            train_metrics = self._train_one_epoch(epoch)

            val_metrics: Dict[str, float] = {}
            if (epoch + 1) % self.app_config.train.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch)

            merged_metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {
                f"val_{k}": v for k, v in val_metrics.items()
            }
            self.callbacks.on_epoch_end(self, epoch, merged_metrics)

            if self.accelerator.is_main_process:
                logger.info("epoch={} metrics={}", epoch, merged_metrics)
