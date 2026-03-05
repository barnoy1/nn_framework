from __future__ import annotations

import warnings
from pathlib import Path

from ...config import AppConfig
from .adapter import FrameworkModelAdapter
from .base import CheckpointAdapter, DnGroupConfigurer, ModelBuilder, ModelWrapperAdapter, WrapperComponents
from .module_loader import load_wrapper_module


def _create_legacy_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    module = load_wrapper_module(repo_root, "adapter.py")

    wrapper_factory = getattr(module, "create_model_wrapper", None)
    if callable(wrapper_factory):
        warnings.warn(
            "Legacy wrapper API create_model_wrapper(...) is deprecated; "
            "expose create_wrapper_components(...) in nn_wrapper/adapter.py",
            DeprecationWarning,
            stacklevel=2,
        )
        return wrapper_factory(app_config=app_config, repo_root=repo_root)

    raise AttributeError(
        "Wrapper must expose create_wrapper_components(app_config, repo_root) "
        "or legacy create_model_wrapper(app_config, repo_root)"
    )


def _load_wrapper_components(app_config: AppConfig, repo_root: Path) -> WrapperComponents:
    module = load_wrapper_module(repo_root, "adapter.py")
    components_factory = getattr(module, "create_wrapper_components", None)
    if not callable(components_factory):
        raise AttributeError("Wrapper must expose create_wrapper_components(app_config, repo_root)")

    components = components_factory(app_config=app_config, repo_root=repo_root)
    if not isinstance(components, WrapperComponents):
        raise TypeError(f"create_wrapper_components must return WrapperComponents, got {type(components)!r}")
    return components


def create_model_builder(app_config: AppConfig, repo_root: Path) -> ModelBuilder:
    try:
        return _load_wrapper_components(app_config=app_config, repo_root=repo_root).model_builder
    except (AttributeError, TypeError):
        pass

    module = load_wrapper_module(repo_root, "builder.py")

    builder_factory = getattr(module, "create_model_builder", None)
    if callable(builder_factory):
        warnings.warn(
            "Legacy builder API create_model_builder(...) is deprecated; "
            "expose create_wrapper_components(...) in nn_wrapper/adapter.py",
            DeprecationWarning,
            stacklevel=2,
        )
        return builder_factory(app_config=app_config, repo_root=repo_root)

    raise AttributeError(
        "Wrapper must expose create_wrapper_components(app_config, repo_root) "
        "or legacy create_model_builder(app_config, repo_root)"
    )


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    try:
        components = _load_wrapper_components(app_config=app_config, repo_root=repo_root)
        return FrameworkModelAdapter(
            model_builder=components.model_builder,
            checkpoint_adapter=components.checkpoint_adapter,
            dn_group_configurer=components.dn_group_configurer,
        )
    except (AttributeError, TypeError):
        try:
            return FrameworkModelAdapter(
                model_builder=create_model_builder(app_config=app_config, repo_root=repo_root),
                checkpoint_adapter=create_checkpoint_adapter(repo_root=repo_root),
                dn_group_configurer=create_dn_group_configurer(repo_root=repo_root),
            )
        except Exception:
            return _create_legacy_model_wrapper(app_config=app_config, repo_root=repo_root)


def create_dn_group_configurer(repo_root: Path) -> DnGroupConfigurer:
    module = load_wrapper_module(repo_root, "builder.py")
    configurer = getattr(module, "configure_fixed_dn_num_group", None)
    if not callable(configurer):
        raise AttributeError("Wrapper must expose callable configure_fixed_dn_num_group")
    return configurer


def create_checkpoint_adapter(repo_root: Path) -> CheckpointAdapter:
    try:
        module = load_wrapper_module(repo_root, "adapter.py")
        adapter_factory = getattr(module, "create_checkpoint_adapter", None)
        if callable(adapter_factory):
            return adapter_factory(repo_root=repo_root)
    except Exception:
        pass

    module = load_wrapper_module(repo_root, "checkpoint.py")

    adapter_factory = getattr(module, "create_checkpoint_adapter", None)
    if callable(adapter_factory):
        warnings.warn(
            "Legacy checkpoint API create_checkpoint_adapter(...) is deprecated; "
            "expose create_wrapper_components(...) in nn_wrapper/adapter.py",
            DeprecationWarning,
            stacklevel=2,
        )
        return adapter_factory(repo_root=repo_root)

    raise AttributeError(
        "Wrapper must expose create_wrapper_components(app_config, repo_root) "
        "or legacy create_checkpoint_adapter(repo_root)"
    )
