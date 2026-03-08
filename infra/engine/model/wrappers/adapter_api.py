from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ....config import AppConfig
from .contracts import ModelBuilder, WrapperComponents


class AdapterIntegrationAPI(Protocol):
    """Framework-side contract that concrete nn_wrapper integrations should implement.

    Required public function:
    - create_wrapper_components(app_config, repo_root) -> WrapperComponents

    Optional compatibility function:
    - create_model_builder(app_config, repo_root) -> ModelBuilder
    """

    def create_wrapper_components(self, app_config: AppConfig, repo_root: Path) -> WrapperComponents:
        ...

    def create_model_builder(self, app_config: AppConfig, repo_root: Path) -> ModelBuilder:
        ...


REQUIRED_PUBLIC_FUNCTIONS = (
    "create_wrapper_components",
)

OPTIONAL_PUBLIC_FUNCTIONS = (
    "create_model_builder",
)
