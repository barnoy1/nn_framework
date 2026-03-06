from __future__ import annotations

import gc
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from .callbacks import CallbackList
from ..config import AppConfig
from .model import ModelWrapperAdapter
from .training import (
    LossComponentSplitter,
)
from .trainer_epoch import train_one_epoch
from .trainer_eval import run_baseline_eval_sanity, validate_epoch
from .trainer_utils import compute_validation_loss_components_for_trainer, split_loss_components
from infra.utils.log.logger import logger

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
        self.logger = logger_port or logger

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
        self._warned_unmatched_configured_losses = False
        self.last_validation_confusion_matrix: Optional[np.ndarray] = None
        self.last_validation_confusion_labels: list[str] = []
        self.total_epochs = (
            int(self.app_config.runtime.epoches)
            if self.app_config.runtime.epoches is not None
            else int(self.app_config.train.epochs)
        )
        self._loss_splitter = LossComponentSplitter.from_config(self.app_config)

    def _split_loss_components(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return split_loss_components(self, loss_dict)

    @torch.no_grad()
    def _compute_validation_loss_components(self) -> Dict[str, float]:
        return compute_validation_loss_components_for_trainer(self)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        return train_one_epoch(self, epoch)

    def _cleanup_gpu_memory(self, epoch: int) -> None:
        if not torch.cuda.is_available():
            return

        device = self.accelerator.device
        if not str(device).startswith("cuda"):
            return

        self.accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        if self.accelerator.is_main_process:
            self.logger.debug("Cleared CUDA cache at end of epoch={}", epoch)

    @torch.no_grad()
    def validate(self, epoch: int, score_thr: Optional[float] = None) -> Dict[str, float]:
        return validate_epoch(self, epoch, score_thr=score_thr)

    @torch.no_grad()
    def run_baseline_eval_sanity(self, epoch: int = -1, score_thr: Optional[float] = None) -> Dict[str, float]:
        return run_baseline_eval_sanity(self, epoch=epoch, score_thr=score_thr)

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

                self._cleanup_gpu_memory(epoch)
        finally:
            self.callbacks.on_train_end(self)
