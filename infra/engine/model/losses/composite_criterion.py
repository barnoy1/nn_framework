from __future__ import annotations

from typing import Dict, Iterable

from torch import nn


class CompositeCriterion(nn.Module):
    def __init__(self, base_criterion: nn.Module, adapters: Iterable, resolver) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.adapters = list(adapters)
        self.resolver = resolver

    def _weight_dict(self) -> Dict[str, float]:
        payload = getattr(self.base_criterion, "weight_dict", {})
        if not isinstance(payload, dict):
            return {}
        return {str(key).strip().lower(): float(value) for key, value in payload.items()}

    def forward(self, outputs, targets, **kwargs):
        base_loss_dict = self.base_criterion(outputs, targets, **kwargs)
        if not isinstance(base_loss_dict, dict) or not base_loss_dict:
            return base_loss_dict

        default_weight_dict = self._weight_dict()
        merged = dict(base_loss_dict)

        for adapter in self.adapters:
            adapted_losses = adapter.transform(
                loss_dict=base_loss_dict,
                default_weight_dict=default_weight_dict,
                resolver=self.resolver,
            )
            if not adapted_losses:
                continue
            merged.update(adapted_losses)

        return merged
