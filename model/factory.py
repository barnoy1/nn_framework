from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from ..config import AppConfig
from .base import CheckpointAdapter, DnGroupConfigurer, ModelBuilder, ModelWrapperAdapter


def _load_wrapper_module(repo_root: Path, module_file: str) -> ModuleType:
    module_path = repo_root / "nn_wrapper" / module_file
    if not module_path.exists():
        raise FileNotFoundError(f"Wrapper module not found: {module_path}")

    module_name = f"nn_framework_nn_wrapper_{abs(hash(module_path.resolve()))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    module = _load_wrapper_module(repo_root, "adapter.py")

    wrapper_factory = getattr(module, "create_model_wrapper", None)
    if callable(wrapper_factory):
        return wrapper_factory(app_config=app_config, repo_root=repo_root)

    wrapper_cls = getattr(module, "RTDETRv2WrapperAdapter", None)
    if wrapper_cls is None:
        raise AttributeError(
            "Wrapper must expose create_model_wrapper(app_config, repo_root) "
            "or RTDETRv2WrapperAdapter"
        )
    return wrapper_cls(app_config=app_config, repo_root=repo_root)


def create_model_builder(app_config: AppConfig, repo_root: Path) -> ModelBuilder:
    wrapper = create_model_wrapper(app_config=app_config, repo_root=repo_root)

    class _BuilderShim(ModelBuilder):
        def build(self):
            return wrapper.build_components()

    return _BuilderShim()


def create_dn_group_configurer(repo_root: Path) -> DnGroupConfigurer:
    module = _load_wrapper_module(repo_root, "builder.py")
    configurer = getattr(module, "configure_fixed_dn_num_group", None)
    if not callable(configurer):
        raise AttributeError("Wrapper must expose callable configure_fixed_dn_num_group")
    return configurer


def create_checkpoint_adapter(repo_root: Path) -> CheckpointAdapter:
    module = _load_wrapper_module(repo_root, "checkpoint.py")

    adapter_factory = getattr(module, "create_checkpoint_adapter", None)
    if callable(adapter_factory):
        return adapter_factory(repo_root=repo_root)

    adapter_cls = getattr(module, "RTDETRCheckpointAdapter", None)
    if adapter_cls is None:
        raise AttributeError(
            "Wrapper must expose create_checkpoint_adapter(repo_root) "
            "or RTDETRCheckpointAdapter"
        )
    return adapter_cls(repo_root=repo_root)
