from __future__ import annotations

import sys
import tempfile
from importlib import import_module
from pathlib import Path

from torch import nn
import yaml

from infra.engine.model.wrappers.common import AgnosticModelBuilderBase


class RTDETRv2ModelBuilder(AgnosticModelBuilderBase):
    _REPO_ROOT_TOKEN = "@REPO_ROOT/"

    def __init__(self, app_config, repo_root: Path) -> None:
        super().__init__(app_config=app_config, repo_root=repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        self._adapter_root = Path(__file__).resolve().parent
        self._workspace_root = self.repo_root.parents[2]

    def _resolve_model_config_path(self) -> Path:
        configured_path = str(self.app_config.model.model_config_path).strip()
        if not configured_path:
            raise ValueError("model.model_config_path must not be empty")

        candidate = Path(configured_path).expanduser()
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            raise FileNotFoundError(f"Model config not found: {candidate}")

        repo_relative = (self.repo_root / candidate).resolve()
        if repo_relative.exists():
            return repo_relative

        adapter_relative = (self._adapter_root / candidate).resolve()
        if adapter_relative.exists():
            return adapter_relative

        adapter_by_name = (self._adapter_root / "configs" / "rtdetrv2" / candidate.name).resolve()
        if adapter_by_name.exists():
            return adapter_by_name

        raise FileNotFoundError(
            "Model config not found in model repo or adapter configs: "
            f"configured={configured_path}, repo_candidate={repo_relative}, adapter_candidate={adapter_relative}"
        )

    def _load_model_config(self):
        core_module = import_module("src.core")
        yaml_config_cls = getattr(core_module, "YAMLConfig")
        config_path = self._resolve_model_config_path()
        if self._adapter_root in config_path.parents:
            config_path = self._materialize_repo_root_includes(config_path)
        return yaml_config_cls(str(config_path))

    def _materialize_repo_root_includes(self, config_path: Path) -> Path:
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        includes = payload.get("__include__")
        if not isinstance(includes, list):
            return config_path

        resolved_includes: list[str] = []
        changed = False
        for include_path in includes:
            include_text = str(include_path)
            if include_text.startswith(self._REPO_ROOT_TOKEN):
                relative_to_root = include_text[len(self._REPO_ROOT_TOKEN) :]
                absolute = (self._workspace_root / relative_to_root).resolve()
                resolved_includes.append(str(absolute))
                changed = True
                continue
            resolved_includes.append(include_text)

        if not changed:
            return config_path

        payload["__include__"] = resolved_includes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", encoding="utf-8", delete=False) as temp_file:
            yaml.safe_dump(payload, temp_file, sort_keys=False)
            return Path(temp_file.name)

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        yaml_cfg = self._load_model_config()
        return yaml_cfg.model, yaml_cfg.criterion, yaml_cfg.postprocessor
