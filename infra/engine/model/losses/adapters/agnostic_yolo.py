from __future__ import annotations

from typing import Dict

from ..base import YoloLossAdapterBase


class AgnosticYoloCriterionAdapter(YoloLossAdapterBase):
    name = "common"

    _common_base_terms = (
        "loss_bbox",
        "loss_giou",
        "loss_vfl",
        "loss_focal",
        "loss_dfl",
    )

    def owns(self, loss_key: str, resolver=None) -> bool:
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
            transformed[key] = self._scale_value(
                key_text=key_text,
                value=value,
                default_weight_dict=default_weight_dict,
                resolver=resolver,
            )
        return transformed
