from __future__ import annotations

from ..runtime import (
    load_partial_pretrained_weights,
    maybe_download_pretrain_weights,
    resolve_pretrain_weights_path,
)


class RFDETRWeightsOverride:
    def apply(self, *, builder, state) -> None:
        model_factory = state.extras["model_factory"]
        config_payload = state.config_payload or {}
        num_channels = int(config_payload.get("num_channels", 3) or 3)

        if num_channels == 1:
            partial_pretrain_path = resolve_pretrain_weights_path(state.model_config)
            state.model_config.pretrain_weights = None
            state.model_api = model_factory(**state.model_config.model_dump())
            if partial_pretrain_path:
                load_partial_pretrained_weights(
                    model=state.model_api.model,
                    checkpoint_path=partial_pretrain_path,
                )
        else:
            maybe_download_pretrain_weights(state.model_config)
            state.model_api = model_factory(**state.model_config.model_dump())

        state.runtime_args = state.model_api.args
