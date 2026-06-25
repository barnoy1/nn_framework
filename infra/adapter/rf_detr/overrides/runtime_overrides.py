from __future__ import annotations


class RFDETRRuntimeOverride:
    def apply(self, *, builder, state) -> None:
        from rfdetr.main import Model
        from rfdetr.models import build_criterion_and_postprocessors

        state.model_factory = Model
        state.criterion_factory = build_criterion_and_postprocessors
