from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_import_paths(repo_root: Path) -> None:
    for import_path in (repo_root, repo_root / "src"):
        path_text = str(import_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def load_yaml_config(config_path: Path):
    from src.core import YAMLConfig

    return YAMLConfig(str(config_path))