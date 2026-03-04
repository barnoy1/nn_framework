from __future__ import annotations

from pathlib import Path

from infra.config import AppConfig
from infra.engine.model import ModelWrapperAdapter, create_model_wrapper

from .config_loader import REPO_ROOT


def resolve_model_root(config: AppConfig) -> Path:
    candidate = Path(config.model.source_root).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def create_wrapper(config: AppConfig) -> ModelWrapperAdapter:
    model_root = resolve_model_root(config)
    return create_model_wrapper(config, repo_root=model_root)
