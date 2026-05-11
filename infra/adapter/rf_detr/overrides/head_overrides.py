from __future__ import annotations


def _realign_head_to_config(*, model, runtime_args) -> None:
    configured = int(getattr(runtime_args, "num_classes", 0)) + 1
    if configured <= 1:
        return
    class_embed = getattr(model, "class_embed", None)
    if class_embed is None:
        return
    actual = class_embed.bias.shape[0]
    if actual != configured:
        model.reinitialize_detection_head(configured)


class RFDETRHeadOverride:
    def apply(self, *, builder, state) -> None:
        criterion_factory = state.extras["criterion_factory"]
        model = state.model_api.model
        _realign_head_to_config(model=model, runtime_args=state.runtime_args)
        criterion, _ = criterion_factory(state.runtime_args)
        state.model = model
        state.criterion = criterion
        state.postprocessor = state.model_api.postprocess
