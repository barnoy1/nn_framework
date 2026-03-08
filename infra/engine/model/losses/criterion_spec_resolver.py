from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from infra.common.loss_aliases import canonical_loss_alias

from .contracts import ResolvedLossTarget


@dataclass(frozen=True)
class ConfiguredLossSpec:
    pattern: str
    coef: Optional[float]


class DualCriterionSpecResolver:
    def __init__(
        self,
        *,
        adapter_common_specs: Iterable[ConfiguredLossSpec],
        concrete_specs: Iterable[ConfiguredLossSpec],
        fallback_to_model_default: bool,
    ) -> None:
        self._adapter_common_specs = [item for item in adapter_common_specs]
        self._concrete_specs = [item for item in concrete_specs]
        self._fallback_to_model_default = bool(fallback_to_model_default)

    @classmethod
    def from_app_config(cls, app_config) -> "DualCriterionSpecResolver":
        pairs = app_config.model.losses.criterion_pairs

        common_specs = [
            ConfiguredLossSpec(pattern=canonical_loss_alias(str(item.loss)), coef=item.coef)
            for item in pairs.iter_adapter_common()
        ]
        concrete_specs = [
            ConfiguredLossSpec(pattern=canonical_loss_alias(str(item.loss)), coef=item.coef)
            for item in pairs.iter_concrete_model()
        ]

        return cls(
            adapter_common_specs=common_specs,
            concrete_specs=concrete_specs,
            fallback_to_model_default=bool(getattr(app_config.model.losses, "fallback_to_model_default", True)),
        )

    @staticmethod
    def _base_loss_key(loss_key: str) -> str:
        lowered = str(loss_key).strip().lower()
        for marker in ("_aux_", "_dn_", "_enc_"):
            marker_index = lowered.find(marker)
            if marker_index > 0:
                return lowered[:marker_index]
        return lowered

    @staticmethod
    def _matches(loss_key: str, pattern: str) -> bool:
        normalized_pattern = str(pattern).strip().lower()
        if not normalized_pattern:
            return False
        lowered_key = str(loss_key).strip().lower()
        if normalized_pattern.endswith("_"):
            return lowered_key.startswith(normalized_pattern)
        return lowered_key == normalized_pattern or lowered_key.startswith(f"{normalized_pattern}_")

    @classmethod
    def _resolve_from_specs(cls, loss_key: str, specs: Iterable[ConfiguredLossSpec]) -> Optional[float]:
        best_match_len = -1
        resolved: Optional[float] = None
        for item in specs:
            if item.coef is None:
                continue
            if not cls._matches(loss_key, item.pattern):
                continue
            if len(item.pattern) > best_match_len:
                best_match_len = len(item.pattern)
                resolved = float(item.coef)
        return resolved

    @classmethod
    def _resolve_default_coef(cls, loss_key: str, default_weight_dict: Dict[str, float]) -> float:
        lowered_key = str(loss_key).strip().lower()
        exact = default_weight_dict.get(lowered_key)
        if exact is not None:
            return float(exact)

        base_key = cls._base_loss_key(lowered_key)
        if base_key in default_weight_dict:
            return float(default_weight_dict[base_key])

        return 1.0

    def has_explicit_concrete_match(self, loss_key: str) -> bool:
        return self._resolve_from_specs(loss_key, self._concrete_specs) is not None

    def resolve(self, loss_key: str, default_weight_dict: Dict[str, float]) -> ResolvedLossTarget:
        concrete = self._resolve_from_specs(loss_key, self._concrete_specs)
        if concrete is not None:
            return ResolvedLossTarget(coef=float(concrete), source="concrete")

        common = self._resolve_from_specs(loss_key, self._adapter_common_specs)
        if common is not None:
            return ResolvedLossTarget(coef=float(common), source="common")

        if self._fallback_to_model_default:
            return ResolvedLossTarget(
                coef=self._resolve_default_coef(loss_key, default_weight_dict),
                source="default",
            )

        return ResolvedLossTarget(coef=0.0, source="none")
