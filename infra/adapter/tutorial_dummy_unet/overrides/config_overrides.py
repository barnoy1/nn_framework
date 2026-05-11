from __future__ import annotations

from ..runtime import (
    apply_single_channel_adapter_policy,
    ensure_repo_import_paths,
    load_tutorial_payload,
)


class DummyConfigOverride:
    def apply(self, *, builder, state) -> None:
        ensure_repo_import_paths(builder.repo_root)
        loaded_payload = load_tutorial_payload(state.runtime_config_path)
        state.config_payload = apply_single_channel_adapter_policy(loaded_payload)
