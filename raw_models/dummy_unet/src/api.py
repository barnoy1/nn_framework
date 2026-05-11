from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch import nn

from .simple_unet import SimpleUNet


@dataclass(frozen=True)
class DummyUNetConfig:
    in_channels: int = 3
    num_classes: int = 2
    base_channels: int = 16

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DummyUNetConfig":
        return cls(
            in_channels=int(payload.get("in_channels", 3)),
            num_classes=int(payload.get("num_classes", 2)),
            base_channels=int(payload.get("base_channels", 16)),
        )


class DummySegCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        logits = outputs["pred_masks"]
        if not targets:
            return {"loss_ce": logits.mean() * 0.0}
        stacked = torch.stack([target["mask"].long() for target in targets], dim=0)
        return {"loss_ce": self._loss(logits, stacked)}


class DummySegPostprocessor(nn.Module):
    def forward(self, outputs: dict[str, torch.Tensor], _sizes: torch.Tensor | None = None) -> list[dict[str, torch.Tensor]]:
        masks = torch.argmax(outputs["pred_masks"], dim=1)
        return [{"mask": mask} for mask in masks]


@dataclass
class DummyUNetRuntimeAPI:
    model: nn.Module
    criterion: nn.Module
    postprocessor: nn.Module
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DummyUNetRuntimeAPI":
        config = DummyUNetConfig.from_payload(payload)
        model = SimpleUNet(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
        )
        return cls(
            model=model,
            criterion=DummySegCriterion(),
            postprocessor=DummySegPostprocessor(),
            metadata={
                "requested_in_channels": int(payload.get("requested_in_channels", 3)),
                "effective_model_in_channels": int(payload.get("in_channels", 3)),
                "adapter_channel_policy": str(
                    payload.get("adapter_channel_policy", "none")
                ),
            },
        )
