from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from infra.common import RuntimePathResolver
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
        self._path_resolver = RuntimePathResolver(
            repo_root=repo_root,
            extra_search_roots=extra_search_roots,
        )

    def resolve_checkpoint_path(self, path: str) -> Path:
        return self._path_resolver.resolve_checkpoint(path)

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        checkpoint_path = self.resolve_checkpoint_path(path)
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        except pickle.UnpicklingError:
            logger.warning(
                "Retrying checkpoint load with weights_only=False for {}",
                checkpoint_path,
            )
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
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
