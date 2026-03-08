from __future__ import annotations

import warnings
from pathlib import Path

from ....config import AppConfig
from .adapter_api import OPTIONAL_PUBLIC_FUNCTIONS, REQUIRED_PUBLIC_FUNCTIONS
from .adapter_runtime import FrameworkModelAdapter
from .contracts import ModelBuilder, ModelWrapperAdapter, WrapperComponents
from .module_loader import load_wrapper_module


def _validate_adapter_api(module) -> None:
    missing = [name for name in REQUIRED_PUBLIC_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        required_text = ", ".join(REQUIRED_PUBLIC_FUNCTIONS)
        raise AttributeError(
            "Wrapper adapter API is incomplete. Missing required functions: "
            f"{', '.join(missing)}. "
            f"Required: {required_text}. "
            "See infra/engine/model/wrappers/adapter_api.py"
        )

    for name in OPTIONAL_PUBLIC_FUNCTIONS:
        value = getattr(module, name, None)
        if value is not None and not callable(value):
            raise TypeError(
                f"Optional wrapper API function '{name}' must be callable when present. "
                "See infra/engine/model/wrappers/adapter_api.py"
            )


def _load_wrapper_components(app_config: AppConfig, repo_root: Path) -> WrapperComponents:
    module = load_wrapper_module(repo_root, "adapter.py")
    _validate_adapter_api(module)
    components_factory = getattr(module, "create_wrapper_components", None)

    components = components_factory(app_config=app_config, repo_root=repo_root)
    if not isinstance(components, WrapperComponents):
        raise TypeError(f"create_wrapper_components must return WrapperComponents, got {type(components)!r}")
    return components


def create_model_builder(app_config: AppConfig, repo_root: Path) -> ModelBuilder:
    try:
        return _load_wrapper_components(app_config=app_config, repo_root=repo_root).model_builder
    except (AttributeError, TypeError):
        pass

    module = load_wrapper_module(repo_root, "adapter.py")

    builder_factory = getattr(module, "create_model_builder", None)
    if callable(builder_factory):
        warnings.warn(
            "Legacy builder API create_model_builder(...) is deprecated; "
            "expose create_wrapper_components(...) in nn_wrapper/adapter.py "
            "(contract in infra/engine/model/wrappers/adapter_api.py)",
            DeprecationWarning,
            stacklevel=2,
        )
        return builder_factory(app_config=app_config, repo_root=repo_root)

    raise AttributeError(
        "Wrapper must expose create_wrapper_components(app_config, repo_root) "
        "or legacy create_model_builder(app_config, repo_root). "
        "See infra/engine/model/wrappers/adapter_api.py for expected API."
    )


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    try:
        components = _load_wrapper_components(app_config=app_config, repo_root=repo_root)
        return FrameworkModelAdapter(
            model_builder=components.model_builder,
        )
    except (AttributeError, TypeError):
        return FrameworkModelAdapter(
            model_builder=create_model_builder(app_config=app_config, repo_root=repo_root),
        )
