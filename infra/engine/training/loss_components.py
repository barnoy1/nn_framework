from __future__ import annotations

from typing import Dict

import torch


class LossComponentSplitter:
    def __init__(self, terms: Dict[str, list[str]]) -> None:
        self._terms = terms

    @classmethod
    def from_config(cls, app_config) -> "LossComponentSplitter":
        configured_pairs = app_config.model.losses.criterion_pairs
        terms = {
            "box": [str(item.loss).lower() for item in configured_pairs.box],
            "cls": [str(item.loss).lower() for item in configured_pairs.cls],
            "dfl": [str(item.loss).lower() for item in configured_pairs.dfl],
            "custom": [str(item.loss).lower() for item in configured_pairs.custom],
        }
        return cls(terms=terms)

    @staticmethod
    def _matches_any(loss_key: str, terms: list[str]) -> bool:
        lowered = str(loss_key).lower()
        for term in terms:
            normalized = str(term).strip().lower()
            if not normalized:
                continue
            if normalized in lowered:
                return True
        return False

    def split(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        box_loss = 0.0
        cls_loss = 0.0
        dfl_loss = 0.0
        custom_loss = 0.0

        for key, value in loss_dict.items():
            if value is None:
                continue
            numeric = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            if self._matches_any(str(key), self._terms["custom"]):
                custom_loss += numeric
            elif self._matches_any(str(key), self._terms["dfl"]):
                dfl_loss += numeric
            elif self._matches_any(str(key), self._terms["box"]):
                box_loss += numeric
            elif self._matches_any(str(key), self._terms["cls"]):
                cls_loss += numeric

        return {
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "dfl_loss": dfl_loss,
            "custom_loss": custom_loss,
        }
