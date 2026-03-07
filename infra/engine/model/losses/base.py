from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass(frozen=True)
class ResolvedLossTarget:
    coef: float
    source: str


class CriterionSpecResolver(Protocol):
    def resolve(self, loss_key: str, default_weight_dict: Dict[str, float]) -> ResolvedLossTarget:
        ...


class DFLossProvider(Protocol):
    def __call__(self, *, outputs, targets, indices, num_boxes: float) -> Dict[str, torch.Tensor]:
        ...


class LossCriterionAdapter(Protocol):
    name: str

    def owns(self, loss_key: str, resolver: Optional[CriterionSpecResolver] = None) -> bool:
        ...

    def forward(
        self,
        loss_dict,
        default_weight_dict: Dict[str, float],
        resolver: CriterionSpecResolver,
    ) -> Dict[str, object]:
        ...

    def get_loss(
        self,
        loss_name: str,
        loss_dict,
        default_weight_dict: Dict[str, float],
        resolver: CriterionSpecResolver,
    ) -> Dict[str, object]:
        ...

    def transform(
        self,
        loss_dict,
        default_weight_dict: Dict[str, float],
        resolver: CriterionSpecResolver,
    ) -> Dict[str, object]:
        ...


class YoloLossAdapterBase:
    _variant_markers = ("_aux_", "_dn_", "_enc_")

    @classmethod
    def _base_key(cls, loss_key: str) -> str:
        lowered = str(loss_key).strip().lower()
        for marker in cls._variant_markers:
            marker_index = lowered.find(marker)
            if marker_index > 0:
                return lowered[:marker_index]
        return lowered

    def _scale_value(self, key_text: str, value, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        target = resolver.resolve(key_text, default_weight_dict)
        base_coef = float(default_weight_dict.get(key_text, default_weight_dict.get(self._base_key(key_text), 1.0)))
        scale = 0.0 if base_coef == 0.0 else float(target.coef) / base_coef

        if torch.is_tensor(value):
            return value * scale
        return float(value) * scale


class ModelAgnosticDetCriterionAdapterBase(YoloLossAdapterBase):
    """Model-agnostic adapter API aligned with DetCriterion structure.

    The adapter consumes criterion-produced losses and applies deterministic
    key-based ownership + scaling so weighted terms remain differentiable.
    """

    losses = ("boxes", "vfl", "focal", "dfl")

    def transform(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver) -> Dict[str, object]:
        return self.forward(loss_dict=loss_dict, default_weight_dict=default_weight_dict, resolver=resolver)

    def forward(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver) -> Dict[str, object]:
        merged: Dict[str, object] = {}
        for loss_name in self.losses:
            merged.update(
                self.get_loss(
                    loss_name=loss_name,
                    loss_dict=loss_dict,
                    default_weight_dict=default_weight_dict,
                    resolver=resolver,
                )
            )
        return merged

    def get_loss(self, loss_name: str, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        loss_map = {
            "boxes": self.loss_boxes,
            "vfl": self.loss_labels_vfl,
            "focal": self.loss_labels_focal,
            "dfl": self.loss_dfl,
        }
        if loss_name not in loss_map:
            raise ValueError(f"Unsupported model-agnostic loss '{loss_name}'")
        return loss_map[loss_name](loss_dict, default_weight_dict, resolver)

    def _collect_scaled(
        self,
        *,
        loss_dict,
        default_weight_dict: Dict[str, float],
        resolver: CriterionSpecResolver,
        base_terms: tuple[str, ...],
    ) -> Dict[str, object]:
        collected: Dict[str, object] = {}
        for key, value in loss_dict.items():
            key_text = str(key).strip().lower()
            if self._base_key(key_text) not in base_terms:
                continue
            if not self.owns(key_text, resolver):
                continue
            collected[str(key)] = self._scale_value(
                key_text=key_text,
                value=value,
                default_weight_dict=default_weight_dict,
                resolver=resolver,
            )
        return collected

    def loss_boxes(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        return self._collect_scaled(
            loss_dict=loss_dict,
            default_weight_dict=default_weight_dict,
            resolver=resolver,
            base_terms=("loss_bbox", "loss_giou"),
        )

    def loss_labels_vfl(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        return self._collect_scaled(
            loss_dict=loss_dict,
            default_weight_dict=default_weight_dict,
            resolver=resolver,
            base_terms=("loss_vfl",),
        )

    def loss_labels_focal(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        return self._collect_scaled(
            loss_dict=loss_dict,
            default_weight_dict=default_weight_dict,
            resolver=resolver,
            base_terms=("loss_focal",),
        )

    def loss_dfl(self, loss_dict, default_weight_dict: Dict[str, float], resolver: CriterionSpecResolver):
        return self._collect_scaled(
            loss_dict=loss_dict,
            default_weight_dict=default_weight_dict,
            resolver=resolver,
            base_terms=("loss_dfl",),
        )
