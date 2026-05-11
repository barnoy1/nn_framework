from __future__ import annotations

from ..runtime import prepare_weights_policy


class RTDETRv2WeightsOverride:
    def apply(self, *, builder, state) -> None:
        state.runtime_config_path = prepare_weights_policy(
            config_path=state.runtime_config_path
        )
