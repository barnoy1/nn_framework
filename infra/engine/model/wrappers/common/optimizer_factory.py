from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import nn


class BackboneGroupedAdamWFactory:
    def __init__(
        self,
        *,
        lr: float,
        weight_decay: float,
        epochs: int,
        backbone_lr_multiplier: float = 1.0,
        backbone_name_matcher: Callable[[str], bool] | None = None,
        eta_min_ratio: float = 0.01,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.backbone_name_matcher = backbone_name_matcher or (
            lambda name: "backbone" in name
        )
        self.eta_min_ratio = eta_min_ratio

    def split_trainable_params(
        self, model: nn.Module
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        backbone_params: List[nn.Parameter] = []
        non_backbone_params: List[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self.backbone_name_matcher(name):
                backbone_params.append(parameter)
            else:
                non_backbone_params.append(parameter)
        return backbone_params, non_backbone_params

    def build(self, model: nn.Module):
        backbone_params, non_backbone_params = self.split_trainable_params(model)
        backbone_lr = self.lr * self.backbone_lr_multiplier

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": non_backbone_params, "lr": self.lr},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * self.eta_min_ratio,
        )
        return optimizer, scheduler
