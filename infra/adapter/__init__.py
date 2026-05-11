from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_model_builder(*, app_config: Any, repo_root: Path):
    from .core.registry import resolve_model_builder as _resolve_model_builder

    return _resolve_model_builder(app_config=app_config, repo_root=repo_root)


def _get_registered_adapters():
    from .core.registry import REGISTERED_ADAPTERS

    return REGISTERED_ADAPTERS


def __getattr__(name: str):
    if name == "REGISTERED_ADAPTERS":
        return _get_registered_adapters()
    if name in {"AdapterManifest", "AdapterSpec"}:
        from .core.spec import AdapterManifest, AdapterSpec

        return {"AdapterManifest": AdapterManifest, "AdapterSpec": AdapterSpec}[name]
    if name == "RTDETRv2ModelBuilder":
        from .rtdetrv2_pytorch import RTDETRv2ModelBuilder

        return RTDETRv2ModelBuilder
    if name == "RFDETRModelBuilder":
        from .rf_detr import RFDETRModelBuilder

        return RFDETRModelBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RTDETRv2ModelBuilder",
    "RFDETRModelBuilder",
    "AdapterManifest",
    "AdapterSpec",
    "REGISTERED_ADAPTERS",
    "resolve_model_builder",
]
