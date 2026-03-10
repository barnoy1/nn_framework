from __future__ import annotations

from typing import Dict

import torch

from infra.common.loss_aliases import canonical_loss_alias


class LossComponentSplitter:
    def __init__(self, terms: Dict[str, list[str]]) -> None:
        self._terms = terms

    def has_dfl_terms(self) -> bool:
        return any(str(term).strip() for term in self._terms.get("common_dfl", []))

    @classmethod
    def from_config(cls, app_config) -> "LossComponentSplitter":
        configured_pairs = app_config.model.losses.criterion_pairs
        common_terms = list(
            dict.fromkeys(
                [
                    canonical_loss_alias(str(item.loss))
                    for item in configured_pairs.iter_adapter_common()
                ]
            )
        )
        dfl_terms = list(
            dict.fromkeys(
                [canonical_loss_alias(str(item.loss)) for item in configured_pairs.dfl]
            )
        )
        concrete_terms = [
            canonical_loss_alias(str(item.loss))
            for item in configured_pairs.iter_concrete_model()
        ]

        terms = {
            "common": common_terms,
            "common_dfl": [term for term in dfl_terms if term in common_terms],
            "custom": list(dict.fromkeys(concrete_terms)),
        }
        return cls(terms=terms)

    @staticmethod
    def _matches_any(loss_key: str, terms: list[str]) -> bool:
        lowered = str(loss_key).lower()
        for term in terms:
            normalized = str(term).strip().lower()
            if not normalized:
                continue
            if normalized.endswith("_"):
                if lowered.startswith(normalized):
                    return True
            elif lowered == normalized or lowered.startswith(f"{normalized}_"):
                return True
        return False

    def split(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        common_totals = {
            term: 0.0 for term in self._terms.get("common", []) if str(term).strip()
        }
        custom_loss = 0.0

        for key, value in loss_dict.items():
            if value is None:
                continue
            numeric = (
                float(value.detach().item()) if torch.is_tensor(value) else float(value)
            )
            if self._matches_any(str(key), self._terms.get("custom", [])):
                custom_loss += numeric
                continue

            for term in self._terms.get("common", []):
                if self._matches_any(str(key), [term]):
                    common_totals[term] = common_totals.get(term, 0.0) + numeric
                    break

        per_common = {
            f"common_{str(term).rstrip('_')}": float(total)
            for term, total in common_totals.items()
        }

        payload = {"custom_loss": custom_loss}

        return payload | per_common
