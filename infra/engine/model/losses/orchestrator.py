from __future__ import annotations

from typing import Dict, Iterable

from torch import nn

from .adapters import ModelAgnosticDetCriterion


def prepare_base_criterion_for_agnostic_flow(
    base_criterion: nn.Module, resolver
) -> None:
    payload = getattr(base_criterion, "weight_dict", {})
    if not isinstance(payload, dict):
        payload = {}

    normalized_weight_dict = {
        str(key).strip().lower(): float(value) for key, value in payload.items()
    }

    def _enabled(loss_key: str) -> bool:
        return float(resolver.resolve(loss_key, normalized_weight_dict).coef) > 0.0

    wants_boxes = _enabled("loss_bbox") or _enabled("loss_giou")
    wants_vfl = _enabled("loss_vfl")
    wants_focal = _enabled("loss_focal")

    if wants_boxes:
        normalized_weight_dict.setdefault("loss_bbox", 1.0)
        normalized_weight_dict.setdefault("loss_giou", 1.0)
    if wants_vfl:
        normalized_weight_dict.setdefault("loss_vfl", 1.0)
    if wants_focal:
        normalized_weight_dict.setdefault("loss_focal", 1.0)

    base_criterion.weight_dict = normalized_weight_dict


class CompositeCriterion(nn.Module):
    def __init__(
        self, base_criterion: nn.Module, adapters: Iterable, resolver, dfl_provider=None
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.adapters = list(adapters)
        self.resolver = resolver
        concrete_probe = next(
            (
                adapter
                for adapter in self.adapters
                if getattr(adapter, "name", "") == "model_specific"
            ),
            None,
        )
        self.model_agnostic_criterion = ModelAgnosticDetCriterion.from_base(
            base_criterion,
            dfl_provider=dfl_provider,
            capability_probe=concrete_probe,
        )

    def _accumulate_concrete_losses(
        self, *, base_loss_dict, default_weight_dict: Dict[str, float]
    ) -> Dict[str, object]:
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

    def _accumulate_common_losses(
        self, *, merged, outputs, targets, default_weight_dict: Dict[str, float]
    ) -> None:
        if self.model_agnostic_criterion is None:
            return
        agnostic_losses = self.model_agnostic_criterion.forward(
            outputs,
            targets,
            resolver=self.resolver,
            default_weight_dict=default_weight_dict,
        )
        for key, value in agnostic_losses.items():
            merged[key] = value

    def _weight_dict(self) -> Dict[str, float]:
        payload = getattr(self.base_criterion, "weight_dict", {})
        if not isinstance(payload, dict):
            return {}
        return {
            str(key).strip().lower(): float(value) for key, value in payload.items()
        }

    def forward_stages(self, outputs, targets, **kwargs):
        base_loss_dict = self.base_criterion(outputs, targets, **kwargs)
        if not isinstance(base_loss_dict, dict):
            return {}, {}

        default_weight_dict = self._weight_dict()
        concrete_losses = self._accumulate_concrete_losses(
            base_loss_dict=base_loss_dict,
            default_weight_dict=default_weight_dict,
        )

        common_losses: Dict[str, object] = {}
        self._accumulate_common_losses(
            merged=common_losses,
            outputs=outputs,
            targets=targets,
            default_weight_dict=default_weight_dict,
        )

        if common_losses or concrete_losses:
            return common_losses, concrete_losses
        return {}, base_loss_dict

    def forward(self, outputs, targets, **kwargs):
        common_losses, concrete_losses = self.forward_stages(outputs, targets, **kwargs)
        merged = dict(concrete_losses)
        merged.update(common_losses)
        return merged
