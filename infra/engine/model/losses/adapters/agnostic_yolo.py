from __future__ import annotations

from ..base import ModelAgnosticDetCriterionAdapterBase


class AgnosticYoloCriterionAdapter(ModelAgnosticDetCriterionAdapterBase):
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
