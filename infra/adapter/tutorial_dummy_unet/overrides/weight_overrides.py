from __future__ import annotations

from ..runtime import maybe_load_checkpoint


class DummyWeightsOverride:
    def apply(self, *, builder, state) -> None:
        model_payload = state.config_payload or {}
        state.model_api = state.model_factory(model_payload)
        maybe_load_checkpoint(
            runtime_api=state.model_api,
            model_payload=model_payload,
        )
