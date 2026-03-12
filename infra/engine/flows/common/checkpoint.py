from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from infra.common import RuntimePathResolver
from infra.common.logging import logger


_CHECKPOINT_PATH_RESOLVER = RuntimePathResolver(
    repo_root=Path(__file__).resolve().parents[3]
)


def resolve_checkpoint_path(path: str) -> Path:
    return _CHECKPOINT_PATH_RESOLVER.resolve_checkpoint(path)


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = resolve_checkpoint_path(path)
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


def _normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def get_checkpoint_num_classes(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    state_dict = _normalize_state_dict_keys(state_dict)
    weight = state_dict.get("decoder.enc_score_head.weight")
    if weight is None:
        return None
    if weight.ndim == 0:
        return None
    return int(weight.shape[0])


def get_model_num_classes(model: nn.Module) -> Optional[int]:
    weight = model.state_dict().get("decoder.enc_score_head.weight")
    if weight is None:
        return None
    if weight.ndim == 0:
        return None
    return int(weight.shape[0])


def validate_checkpoint_class_compatibility(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    checkpoint_classes = get_checkpoint_num_classes(state_dict)
    model_classes = get_model_num_classes(model)

    if checkpoint_classes is None or model_classes is None:
        return

    if checkpoint_classes == model_classes:
        return

    message = (
        "Checkpoint/model class mismatch detected: "
        f"checkpoint classes={checkpoint_classes}, model classes={model_classes}. "
        "This leaves detection score heads uninitialized and usually produces very low-confidence detections. "
        "Use a checkpoint trained with the same num_classes as inference config."
    )
    raise RuntimeError(message)


def safe_load_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Tuple[int, int, int]:
    state_dict = _normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()

    compatible: Dict[str, torch.Tensor] = {}
    skipped_shape = 0
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        compatible[key] = value

    missing_keys, _ = model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped_shape, len(missing_keys)
