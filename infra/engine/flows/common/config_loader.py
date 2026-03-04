from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from infra.config import AppConfig

INFRA_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = INFRA_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_app_config(model_profile: str, overrides: List[str], config_path: str) -> AppConfig:
    config_dir = INFRA_ROOT / "config" / "hydra"

    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        relative = config_file.relative_to(config_dir)
    except ValueError as error:
        raise ValueError(f"Config file must be under {config_dir}: {config_file}") from error

    config_name = str(relative.with_suffix("")).replace("\\", "/")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    payload = OmegaConf.to_container(cfg, resolve=True)
    return AppConfig.model_validate(payload)
