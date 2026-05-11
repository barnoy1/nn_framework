from __future__ import annotations

from ..runtime import load_model_components


class RTDETRv2HeadOverride:
    def apply(self, *, builder, state) -> None:
        model, criterion, postprocessor = load_model_components(
            config_path=state.runtime_config_path
        )
        state.model = model
        state.criterion = criterion
        state.postprocessor = postprocessor
