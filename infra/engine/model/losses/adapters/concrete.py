from __future__ import annotations

from typing import Dict

from .agnostic_yolo import AgnosticYoloCriterionAdapter


class ConcreteCriterionAdapter(AgnosticYoloCriterionAdapter):
    name = "concrete"

    def owns(self, loss_key: str, resolver=None) -> bool:
        lowered = str(loss_key).strip().lower()
        if resolver is not None and resolver.has_explicit_concrete_match(lowered):
            return True
        return any(marker in lowered for marker in self._variant_markers)

    def transform(self, loss_dict, default_weight_dict: Dict[str, float], resolver) -> Dict[str, object]:
        transformed: Dict[str, object] = {}
        for key, value in loss_dict.items():
            key_text = str(key).strip().lower()
            if not self.owns(key_text, resolver):
                continue
            transformed[key] = self._scale_value(
                key_text=key_text,
                value=value,
                default_weight_dict=default_weight_dict,
                resolver=resolver,
            )
        return transformed
