from __future__ import annotations


class RTDETRv2HeadOverride:
    def apply(self, *, builder, state) -> None:
        model, criterion, postprocessor = state.model_factory(
            config_path=state.runtime_config_path
        )
        state.model = model
        state.criterion = criterion
        state.postprocessor = postprocessor
