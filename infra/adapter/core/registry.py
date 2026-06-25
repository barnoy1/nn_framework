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
    name = str(app_config.adapter.name or "").strip()
    by_name: dict[str, AdapterManifest] = {}
    for spec in REGISTERED_ADAPTERS:
        if spec.name in by_name:
            raise ValueError(
                f"Duplicate adapter name registered: {spec.name!r}"
            )
        by_name[spec.name] = spec

    spec = by_name.get(name)
    if spec is None:
        available = ", ".join(sorted(by_name)) or "<none>"
        raise NotImplementedError(
            f"No adapter registered for adapter.name={name!r}. "
            f"Available adapters: {available}"
        )
    return spec.builder_factory(app_config, repo_root)


if __name__ == "__main__":
    # ponytail: smallest check for the name-selection error path; known-name
    # resolution is covered end-to-end by the rf_detr train acceptance gate.
    from types import SimpleNamespace

    try:
        resolve_model_builder(
            app_config=SimpleNamespace(adapter=SimpleNamespace(name="missing")),
            repo_root=Path("."),
        )
        raise AssertionError("unknown adapter.name must raise")
    except NotImplementedError as exc:
        assert "missing" in str(exc)
    print("registry self-check OK")
