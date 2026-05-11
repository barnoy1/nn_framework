from __future__ import annotations


class DummyHeadOverride:
    def apply(self, *, builder, state) -> None:
        state.model = state.model_api.model
        state.criterion = state.model_api.criterion
        state.postprocessor = state.model_api.postprocessor
