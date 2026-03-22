from __future__ import annotations

from pathlib import Path

from torch import nn

from infra.engine.model.wrappers.common import ReflectiveYamlAdapterModelBuilderBase

from .runtime import (
    apply_backbone_policy,
    ensure_repo_import_paths,
    load_model_components,
    prepare_weights_policy,
)
from .schemes import (
    MODEL_REPO_ROOT_TOKEN,
    REPO_ROOT_TOKEN,
    YAML_CLASS_PATCHES,
)


class RTDETRv2ModelBuilder(ReflectiveYamlAdapterModelBuilderBase):
    _REPO_ROOT_TOKEN = REPO_ROOT_TOKEN
    _MODEL_REPO_ROOT_TOKEN = MODEL_REPO_ROOT_TOKEN
    _YAML_CLASS_PATCHES = YAML_CLASS_PATCHES
    _CONFIG_SUBDIR = ("configs", "rtdetrv2")

    def __init__(self, app_config, repo_root: Path) -> None:
        super().__init__(
            app_config=app_config,
            repo_root=repo_root,
            adapter_root=Path(__file__).resolve().parent,
            config_subdir=self._CONFIG_SUBDIR,
        )
        ensure_repo_import_paths(self.repo_root)

    def _resolve_runtime_config_path(self) -> Path:
        config_path = self._resolve_model_config_path()
        return self._materialize_runtime_compatible_config(config_path)

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        config_path = self._resolve_runtime_config_path()
        config_path = apply_backbone_policy(config_path=config_path)
        config_path = prepare_weights_policy(config_path=config_path)
        model, criterion, postprocessor = load_model_components(config_path=config_path)
        return model, criterion, postprocessor
