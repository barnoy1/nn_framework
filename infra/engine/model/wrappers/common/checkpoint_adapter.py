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
                ema = checkpoint["ema"]
                return ema.get("module") or ema.get("ema_model") or ema
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

    _CLASS_HEAD_MARKERS = (
        "class_embed",
        "class_head",
        "score_head",
        "cls_score",
        "classifier",
    )

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        normalized = self._normalize_state_dict_keys(state_dict)
        model_state = model.state_dict()
        for key, ckpt_tensor in normalized.items():
            model_tensor = model_state.get(key)
            if model_tensor is None or model_tensor.shape == ckpt_tensor.shape:
                continue
            if not any(marker in key.lower() for marker in self._CLASS_HEAD_MARKERS):
                continue
            message = (
                f"Checkpoint head class-count mismatch on {key!r}: "
                f"checkpoint {tuple(ckpt_tensor.shape)} vs model {tuple(model_tensor.shape)}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("{} (train flow: proceeding)", message)

    def safe_load_state_dict(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        *,
        strict: bool = False,
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
        if strict and (skipped_shape or missing_keys):
            raise ValueError(
                f"Strict checkpoint load failed: skipped_shape={skipped_shape}, "
                f"missing={len(missing_keys)}. Pass --allow-partial to override."
            )
        return len(compatible), skipped_shape, len(missing_keys)


if __name__ == "__main__":
    # ponytail: checkpoint strictness matrix — train warns, eval/strict raise.
    import tempfile

    model = nn.Linear(4, 3)
    model.class_embed = nn.Linear(4, 3)  # head with 3 classes
    adapter = GenericCheckpointAdapter(repo_root=Path(tempfile.gettempdir()))

    good = {k: v.clone() for k, v in model.state_dict().items()}
    mismatch = dict(good)
    mismatch["class_embed.weight"] = torch.zeros(5, 4)  # 5 classes != 3

    # strict class-compat: eval-style hard error; train-style only warns
    try:
        adapter.validate_checkpoint_class_compatibility(model, mismatch, strict=True)
        raise AssertionError("eval class-count mismatch must raise")
    except ValueError:
        pass
    adapter.validate_checkpoint_class_compatibility(model, mismatch, strict=False)

    # strict load fails fast on shape mismatch; permissive proceeds
    try:
        adapter.safe_load_state_dict(model, mismatch, strict=True)
        raise AssertionError("strict load must raise on shape mismatch")
    except ValueError:
        pass
    loaded, skipped, missing = adapter.safe_load_state_dict(model, mismatch, strict=False)
    assert skipped == 1 and loaded >= 1
    adapter.safe_load_state_dict(model, good, strict=True)  # clean load OK
    print("checkpoint strictness self-check OK")
