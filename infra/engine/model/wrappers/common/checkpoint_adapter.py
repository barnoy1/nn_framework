from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from infra.common.logging import logger
from infra.engine.model.wrappers.contracts import CheckpointAdapter


class GenericCheckpointAdapter(CheckpointAdapter):
    def __init__(
        self,
        *,
        repo_root: Path,
        extra_search_roots: Iterable[Path] = (),
    ) -> None:
        self.repo_root = repo_root
        self.extra_search_roots = tuple(extra_search_roots)

    @staticmethod
    def _expand_runtime_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return raw

        # Support Hydra/OmegaConf env interpolation syntax when unresolved
        # by upstream config loading (e.g. ${oc.env:HOME}, ${env:HOME}).
        pattern = re.compile(r"\$\{(?:oc\.env|env):([^}]+)\}")

        def _replace(match: re.Match[str]) -> str:
            variable_name = match.group(1).strip()
            return os.environ.get(variable_name, match.group(0))

        expanded = pattern.sub(_replace, raw)
        return os.path.expandvars(expanded)

    def resolve_checkpoint_path(self, path: str) -> Path:
        resolved_input = self._expand_runtime_path(path)
        candidate = Path(resolved_input).expanduser()
        if candidate.exists():
            return candidate.resolve()

        candidates = []
        if not candidate.is_absolute():
            candidates.append((self.repo_root / candidate).resolve())

        search_roots = [
            (self.repo_root / "weights").resolve(),
            (self.repo_root.parent / "weights").resolve(),
            *[Path(root).resolve() for root in self.extra_search_roots],
        ]
        candidates.extend((root / candidate.name).resolve() for root in search_roots)

        for fallback in candidates:
            if fallback.exists():
                logger.warning(
                    "checkpoint not found at {}, using {}", candidate, fallback
                )
                return fallback

        checked = "\n  - ".join(str(p) for p in [candidate, *candidates])
        raise FileNotFoundError(f"Checkpoint file not found. Checked:\n  - {checked}")

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        checkpoint_path = self.resolve_checkpoint_path(path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "ema" in checkpoint:
                return checkpoint["ema"].get("module", checkpoint["ema"])
            if "model" in checkpoint:
                return checkpoint["model"]
            return checkpoint
        return checkpoint

    @staticmethod
    def _normalize_state_dict_keys(
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not any(key.startswith("module.") for key in state_dict.keys()):
            return state_dict
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        return

    def safe_load_state_dict(
        self, model: nn.Module, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[int, int, int]:
        normalized = self._normalize_state_dict_keys(state_dict)
        model_state = model.state_dict()

        compatible: Dict[str, torch.Tensor] = {}
        skipped_shape = 0
        for key, value in normalized.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped_shape += 1
                continue
            compatible[key] = value

        missing_keys, _ = model.load_state_dict(compatible, strict=False)
        return len(compatible), skipped_shape, len(missing_keys)
