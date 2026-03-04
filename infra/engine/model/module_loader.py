from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_wrapper_module(repo_root: Path, module_file: str) -> ModuleType:
    module_path = repo_root / "nn_wrapper" / module_file
    if not module_path.exists():
        raise FileNotFoundError(f"Wrapper module not found: {module_path}")

    module_name = f"nn_framework_nn_wrapper_{abs(hash(module_path.resolve()))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
