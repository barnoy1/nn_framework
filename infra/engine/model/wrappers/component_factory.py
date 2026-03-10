from __future__ import annotations

from pathlib import Path

from ....config import AppConfig
from .adapter import RFDETRModelBuilder, RTDETRv2ModelBuilder
from .adapter_runtime import FrameworkModelAdapter
from .contracts import ModelWrapperAdapter


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    source_root = str(app_config.model.source_root).lower()
    if "rtdetrv2_pytorch" in source_root:
        return FrameworkModelAdapter(
            model_builder=RTDETRv2ModelBuilder(
                app_config=app_config, repo_root=repo_root
            ),
        )
    if "rf-detr" in source_root or "rfdetr" in source_root:
        return FrameworkModelAdapter(
            model_builder=RFDETRModelBuilder(
                app_config=app_config, repo_root=repo_root
            ),
        )
    raise NotImplementedError(
        f"No model wrapper adapter registered for source_root={app_config.model.source_root}"
    )
