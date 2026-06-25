from __future__ import annotations

from ..runtime import apply_backbone_policy, load_model_components


class RTDETRv2RuntimeOverride:
    def apply(self, *, builder, state) -> None:
        state.runtime_config_path = apply_backbone_policy(
            config_path=state.runtime_config_path
        )
        # rtdetrv2 builds model+criterion+postprocessor as a unit, so the single
        # combined factory is bound here; criterion_factory stays unused.
        state.model_factory = load_model_components
