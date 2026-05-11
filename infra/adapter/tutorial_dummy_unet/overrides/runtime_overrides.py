from __future__ import annotations

from ..runtime import build_runtime_api


class DummyRuntimeOverride:
    def apply(self, *, builder, state) -> None:
        state.model_api = build_runtime_api(state.config_payload or {})
