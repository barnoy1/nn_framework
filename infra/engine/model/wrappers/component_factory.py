from __future__ import annotations

from pathlib import Path

from ....config import AppConfig
from .adapter_runtime import FrameworkModelAdapter
from .common import AgnosticModelBuilderBase
from .contracts import ModelWrapperAdapter


class GenericModelBuilder(AgnosticModelBuilderBase):
    pass


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    return FrameworkModelAdapter(
        model_builder=GenericModelBuilder(app_config=app_config, repo_root=repo_root),
    )
