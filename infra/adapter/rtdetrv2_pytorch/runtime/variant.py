from __future__ import annotations

from pathlib import Path

from .config import load_yaml_config


def load_model_components(*, config_path: Path):
    yaml_cfg = load_yaml_config(config_path)
    return yaml_cfg.model, yaml_cfg.criterion, yaml_cfg.postprocessor