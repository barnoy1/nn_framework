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


def _resolve_relocated_experiment_path(candidate: Path) -> Path | None:
    parts = candidate.parts
    marker = ("infra", "config", "hydra", "experiment")
    if len(parts) < len(marker):
        return None
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            relocated = (REPO_ROOT / "experiment" / candidate.name).resolve()
            if relocated.exists():
                return relocated
            return None
    return None


def load_app_config(overrides: List[str], config_path: str) -> AppConfig:
    config_dir = INFRA_ROOT / "config" / "hydra"

    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()
    if not config_file.exists():
        relocated = _resolve_relocated_experiment_path(config_file)
        if relocated is not None:
            config_file = relocated
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        relative = config_file.relative_to(config_dir)
    except ValueError:
        cfg = OmegaConf.load(str(config_file))
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        payload = OmegaConf.to_container(cfg, resolve=True)
        return AppConfig.model_validate(payload)

    config_name = str(relative.with_suffix("")).replace("\\", "/")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    payload = OmegaConf.to_container(cfg, resolve=True)
    return AppConfig.model_validate(payload)
