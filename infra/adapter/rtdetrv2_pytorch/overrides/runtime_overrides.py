from __future__ import annotations

from ..runtime import apply_backbone_policy


class RTDETRv2RuntimeOverride:
    def apply(self, *, builder, state) -> None:
        state.runtime_config_path = apply_backbone_policy(
            config_path=state.runtime_config_path
        )
