from __future__ import annotations

from typing import Dict

import torch

from ..base import LossCriterionAdapter


class AgnosticYoloCriterionAdapter(LossCriterionAdapter):
    name = "common"

    _common_base_terms = (
        "loss_bbox",
        "loss_giou",
        "loss_vfl",
        "loss_focal",
        "loss_dfl",
    )
    _variant_markers = ("_aux_", "_dn_", "_enc_")

    @classmethod
    def _base_key(cls, loss_key: str) -> str:
        lowered = str(loss_key).strip().lower()
        for marker in cls._variant_markers:
            marker_index = lowered.find(marker)
            if marker_index > 0:
                return lowered[:marker_index]
        return lowered

    def owns(self, loss_key: str) -> bool:
        lowered = str(loss_key).strip().lower()
        if any(marker in lowered for marker in self._variant_markers):
            return False
        return self._base_key(lowered) in self._common_base_terms

    def transform(self, loss_dict, default_weight_dict: Dict[str, float], resolver) -> Dict[str, object]:
        transformed: Dict[str, object] = {}
        for key, value in loss_dict.items():
            key_text = str(key).strip().lower()
            if not self.owns(key_text):
                continue

            target = resolver.resolve(key_text, default_weight_dict)
            base_coef = float(default_weight_dict.get(key_text, default_weight_dict.get(self._base_key(key_text), 1.0)))
            scale = 0.0 if base_coef == 0.0 else float(target.coef) / base_coef

            if torch.is_tensor(value):
                transformed[key] = value * scale
            else:
                transformed[key] = float(value) * scale
        return transformed
