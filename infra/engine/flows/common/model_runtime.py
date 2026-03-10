from __future__ import annotations

from pathlib import Path

from infra.config import AppConfig
from infra.engine.model import ModelWrapperAdapter, create_model_wrapper

from .config_loader import REPO_ROOT


def resolve_model_root(config: AppConfig) -> Path:
    candidate = Path(config.model.source_root).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(
            "Configured model.source_root does not exist or is not a directory: "
            f"{resolved}"
        )
    return resolved


def create_wrapper(config: AppConfig) -> ModelWrapperAdapter:
    model_root = resolve_model_root(config)
    return create_model_wrapper(config, repo_root=model_root)
