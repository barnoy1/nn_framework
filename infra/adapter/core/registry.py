from __future__ import annotations

from pathlib import Path
from typing import Any

from infra.adapter.rf_detr import RFDETRModelBuilder
from infra.adapter.rtdetrv2_pytorch import RTDETRv2ModelBuilder
from infra.engine.model.wrappers.contracts import ModelBuilder

from .spec import AdapterSpec


REGISTERED_ADAPTERS: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        name="rtdetrv2_pytorch",
        source_root_tokens=("rtdetrv2_pytorch",),
        builder_factory=RTDETRv2ModelBuilder,
    ),
    AdapterSpec(
        name="rf_detr",
        source_root_tokens=("rf-detr", "rfdetr"),
        builder_factory=RFDETRModelBuilder,
    ),
)


def resolve_model_builder(*, app_config: Any, repo_root: Path) -> ModelBuilder:
    source_root = str(app_config.model.source_root or "")
    for spec in REGISTERED_ADAPTERS:
        if spec.matches_source_root(source_root):
            return spec.builder_factory(app_config, repo_root)

    available = ", ".join(spec.name for spec in REGISTERED_ADAPTERS)
    raise NotImplementedError(
        "No model wrapper adapter registered for "
        f"source_root={source_root!r}. Available adapters: {available}"
    )