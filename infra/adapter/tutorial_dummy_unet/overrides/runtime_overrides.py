from __future__ import annotations

from ..runtime import build_runtime_api


class DummyRuntimeOverride:
    def apply(self, *, builder, state) -> None:
        state.model_factory = build_runtime_api
