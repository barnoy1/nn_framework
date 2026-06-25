from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

def ensure_repo_import_paths(repo_root: Path) -> None:
    for import_path in (repo_root, repo_root / "src"):
        path_text = str(import_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def load_dino_config(config_path: Path) -> DictConfig:
    with config_path.open("r", encoding="utf-8") as file:
        return OmegaConf.create(json.load(file))


def infer_model_profile(
    *, config_path: Path, config_payload: DictConfig
) -> dict[str, object]:
    from ..schemes import BASE_MODEL_PROFILE, SMALL_MODEL_PROFILE

    config_name = str(config_path.name).lower()
    patch_size = int(config_payload.patch_size if "patch_size" in config_payload else 14)
    image_size = int(config_payload.image_size if "image_size" in config_payload else 518)

    profile = dict(BASE_MODEL_PROFILE) if "base" in config_name else dict(SMALL_MODEL_PROFILE)
    profile.update(
        {
            "patch_size": patch_size,
            "resolution": image_size,
            "positional_encoding_size": image_size // max(1, patch_size),
        }
    )
    return profile