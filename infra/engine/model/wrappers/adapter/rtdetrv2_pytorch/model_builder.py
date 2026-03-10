from __future__ import annotations

from importlib import import_module
from pathlib import Path

from torch import nn

from .schemes import (
    MODEL_REPO_ROOT_TOKEN,
    REPO_ROOT_TOKEN,
    YAML_CLASS_PATCHES,
)
from infra.engine.model.wrappers.common import ReflectiveYamlAdapterModelBuilderBase


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

    def _load_model_config(self):
        core_module = import_module("src.core")
        yaml_config_cls = getattr(core_module, "YAMLConfig")
        config_path = self._resolve_model_config_path()
        config_path = self._materialize_runtime_compatible_config(config_path)
        return yaml_config_cls(str(config_path))

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        yaml_cfg = self._load_model_config()
        criterion = yaml_cfg.criterion
        return yaml_cfg.model, criterion, yaml_cfg.postprocessor
