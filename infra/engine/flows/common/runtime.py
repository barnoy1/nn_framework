from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import DataLoader

from infra.config import AppConfig, set_active_app_config
from infra.engine.model import BuiltComponents, ModelWrapperAdapter

from .config_loader import load_app_config
from .data_runtime import build_loaders as build_data_loaders, prepare_data_if_needed
from .model_runtime import create_wrapper


@dataclass(frozen=True)
class FlowRuntime:
    app_config: AppConfig
    built: BuiltComponents
    wrapper: ModelWrapperAdapter
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]


def build_flow_runtime(
    overrides: List[str], config_path: str, build_loaders: bool = True
) -> FlowRuntime:
    config = load_app_config(overrides=overrides, config_path=config_path)
    set_active_app_config(config)
    if build_loaders:
        config.ensure_output_dir()
    prepare_data_if_needed(config)

    if build_loaders:
        train_loader, val_loader = build_data_loaders(config)
    else:
        train_loader, val_loader = None, None

    wrapper = create_wrapper(config)
    built = wrapper.build_components()

    return FlowRuntime(
        app_config=config,
        built=built,
        wrapper=wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
    )
