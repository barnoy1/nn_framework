from __future__ import annotations

from ..runtime import maybe_load_checkpoint


class DummyWeightsOverride:
    def apply(self, *, builder, state) -> None:
        maybe_load_checkpoint(
            runtime_api=state.model_api,
            model_payload=state.config_payload or {},
        )
