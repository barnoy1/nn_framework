from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from infra.adapter.rf_detr import RFDETRModelBuilder
from infra.adapter.rtdetrv2_pytorch import RTDETRv2ModelBuilder
from infra.engine.model.wrappers.contracts import ModelBuilder

from .spec import AdapterManifest


def _manifest_from_builder(builder_factory) -> AdapterManifest:
    manifest_factory = getattr(builder_factory, "manifest", None)
    if not callable(manifest_factory):
        raise TypeError(
            f"Adapter builder {builder_factory!r} must expose manifest() classmethod"
        )

    manifest = manifest_factory()
    if not isinstance(manifest, AdapterManifest):
        raise TypeError(
            f"Adapter builder {builder_factory!r} returned invalid manifest: {manifest!r}"
        )

    if manifest.builder_factory is not builder_factory:
        manifest = replace(manifest, builder_factory=builder_factory)
    manifest.validate()
    return manifest


REGISTERED_ADAPTERS: tuple[AdapterManifest, ...] = tuple(
    _manifest_from_builder(builder_factory)
    for builder_factory in (RTDETRv2ModelBuilder, RFDETRModelBuilder)
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
