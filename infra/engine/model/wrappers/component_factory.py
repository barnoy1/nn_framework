from __future__ import annotations

from pathlib import Path

from ....config import AppConfig
from infra.adapter import resolve_model_builder
from .adapter_runtime import FrameworkModelAdapter
from .contracts import ModelWrapperAdapter


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    model_builder = resolve_model_builder(app_config=app_config, repo_root=repo_root)
    return FrameworkModelAdapter(model_builder=model_builder)
