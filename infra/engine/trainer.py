from __future__ import annotations

from typing import Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from ..adapters import LoguruLoggerAdapter
from ..core import move_targets_to_device
from .callbacks import CallbackList
from ..config import AppConfig
from .flows.eval.shared import run_eval_artifacts
from .model import ModelWrapperAdapter
from ..interfaces import LoggerPort


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
        logger_port: Optional[LoggerPort] = None,
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
            self.ema_model.align_to_model(self.accelerator.unwrap_model(self.model))

        self.global_step = 0
        self.current_epoch = 0
        self.total_epochs = (
            int(self.app_config.runtime.epoches)
            if self.app_config.runtime.epoches is not None
            else int(self.app_config.train.epochs)
        )

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        num_steps = 0

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.accelerator.device, non_blocking=True)
            targets = move_targets_to_device(targets, self.accelerator.device)

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
                self.logger.info("epoch={} step={} loss={:.6f}", epoch, step, metrics["train/loss"])

        self.scheduler.step()
        return {"loss": running_loss / max(1, num_steps)}

    @torch.no_grad()
    def validate(self, epoch: int, score_thr: float = 0.0) -> Dict[str, float]:
        self.model.eval()

        use_ema = self.ema_model is not None
        if use_ema:
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                self.ema_model.store(unwrapped)
                self.ema_model.copy_to(unwrapped)
            except RuntimeError as error:
                try:
                    self.ema_model.restore(unwrapped)
                except Exception:
                    pass
                self.logger.warning("EMA copy skipped during validate due to state mismatch: {}", error)
                use_ema = False

        class_id_to_name = {int(key): str(value) for key, value in (self.app_config.data.class_id_to_name or {}).items()}
        metrics = run_eval_artifacts(
            app_config=self.app_config,
            model=self.accelerator.unwrap_model(self.model),
            postprocessor=self.postprocessor,
            device=self.accelerator.device,
            logger=self.logger,
            class_id_to_name=class_id_to_name,
            experiment_name=self.experiment_name,
            vis_samples=int(self.app_config.runtime.visualization.num_samples),
            score_thr=float(score_thr),
            image_epoch_suffix=epoch + 1,
            write_metrics_json=True,
        )

        if use_ema:
            self.ema_model.restore(self.accelerator.unwrap_model(self.model))

        self.callbacks.on_validation_end(self, epoch, metrics)
        return metrics

    def fit(self) -> None:
        self.callbacks.on_train_start(self)
        try:
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
