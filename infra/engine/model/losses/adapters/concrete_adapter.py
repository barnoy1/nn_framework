from __future__ import annotations

from typing import Dict

from .common_adapter import AgnosticYoloCriterionAdapter


class ConcreteCriterionAdapter(AgnosticYoloCriterionAdapter):
    name = "concrete"

    _dfl_output_keys = ("pred_distri", "pred_dist", "pred_distributions", "pred_ltrb_dist", "distributions")
    _dfl_precomputed_keys = ("loss_dfl", "dfl_loss", "loss_distribution_focal", "dfl_logits", "dfl_targets")

    def owns(self, loss_key: str, resolver=None) -> bool:
        lowered = str(loss_key).strip().lower()
        if resolver is not None and resolver.has_explicit_concrete_match(lowered):
            return True
        return any(marker in lowered for marker in self._variant_markers)

    def supports_loss(self, loss_name: str, *, outputs=None, targets=None) -> bool:
        normalized = str(loss_name).strip().lower()
        if normalized != "dfl":
            return True
        if not isinstance(outputs, dict):
            return False
        return any(key in outputs for key in self._dfl_output_keys) or any(
            key in outputs for key in self._dfl_precomputed_keys
        )

    def forward(self, loss_dict, default_weight_dict: Dict[str, float], resolver) -> Dict[str, object]:
        transformed = super().forward(
            loss_dict=loss_dict,
            default_weight_dict=default_weight_dict,
            resolver=resolver,
        )

        for key, value in loss_dict.items():
            key_text = str(key).strip().lower()
            if str(key) in transformed:
                continue
            if not self.owns(key_text, resolver):
                continue
            transformed[str(key)] = self._scale_value(
                key_text=key_text,
                value=value,
                default_weight_dict=default_weight_dict,
                resolver=resolver,
            )

        return transformed
