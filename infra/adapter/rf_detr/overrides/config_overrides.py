from __future__ import annotations

from ..runtime import (
    apply_local_dinov2_config,
    apply_single_channel_backbone_policy,
    build_model_config,
    ensure_repo_import_paths,
    infer_model_profile,
    load_dino_config,
)


class RFDETRConfigOverride:
    def apply(self, *, builder, state) -> None:
        ensure_repo_import_paths(builder.repo_root)
        config_payload = load_dino_config(state.config_path)
        model_profile = infer_model_profile(
            config_path=state.config_path,
            config_payload=config_payload,
        )
        apply_local_dinov2_config(
            config_path=state.config_path,
            model_profile=model_profile,
        )
        apply_single_channel_backbone_policy(config_payload=config_payload)
        state.config_payload = config_payload
        state.model_config = build_model_config(
            app_config=builder.app_config,
            config_path=state.config_path,
        )
