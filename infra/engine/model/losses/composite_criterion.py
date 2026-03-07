from __future__ import annotations

from typing import Dict, Iterable

from torch import nn

from .adapters import ModelAgnosticDetCriterion


class CompositeCriterion(nn.Module):
    def __init__(self, base_criterion: nn.Module, adapters: Iterable, resolver, dfl_provider=None) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.adapters = list(adapters)
        self.resolver = resolver
        concrete_probe = next((adapter for adapter in self.adapters if getattr(adapter, "name", "") == "concrete"), None)
        self.model_agnostic_criterion = ModelAgnosticDetCriterion.from_base(
            base_criterion,
            dfl_provider=dfl_provider,
            capability_probe=concrete_probe,
        )

    def _accumulate_concrete_losses(self, *, base_loss_dict, default_weight_dict: Dict[str, float]) -> Dict[str, object]:
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

    def _accumulate_common_losses(self, *, merged, outputs, targets, default_weight_dict: Dict[str, float]) -> None:
        if self.model_agnostic_criterion is None:
            return
        agnostic_losses = self.model_agnostic_criterion.forward(
            outputs,
            targets,
            resolver=self.resolver,
            default_weight_dict=default_weight_dict,
        )
        for key, value in agnostic_losses.items():
            if key not in merged:
                merged[key] = value

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
        merged = self._accumulate_concrete_losses(
            base_loss_dict=base_loss_dict,
            default_weight_dict=default_weight_dict,
        )
        self._accumulate_common_losses(
            merged=merged,
            outputs=outputs,
            targets=targets,
            default_weight_dict=default_weight_dict,
        )

        return merged
