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


class LossCriterionAdapter(Protocol):
    name: str

    def owns(self, loss_key: str, resolver: Optional[CriterionSpecResolver] = None) -> bool:
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
