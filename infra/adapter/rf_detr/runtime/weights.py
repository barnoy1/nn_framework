from __future__ import annotations

import pickle
from pathlib import Path

import torch


def maybe_download_pretrain_weights(model_config) -> None:
    resolved_path = resolve_pretrain_weights_path(model_config)
    if resolved_path is not None:
        model_config.pretrain_weights = str(resolved_path)


def resolve_pretrain_weights_path(model_config) -> str | None:
    from rfdetr.assets.model_weights import download_pretrain_weights

    pretrain_weights = getattr(model_config, "pretrain_weights", None)
    if pretrain_weights is None:
        return None

    cache_dir = (
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "rf_detr"
    ).expanduser()
    requested_path = cache_dir / Path(str(pretrain_weights)).expanduser()
    if requested_path.exists():
        resolved_path = requested_path
    elif requested_path.is_absolute():
        resolved_path = requested_path
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = cache_dir / requested_path.name

    if not resolved_path.exists():
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        download_pretrain_weights(str(resolved_path))

    return str(resolved_path.resolve())


def load_partial_pretrained_weights(*, model, checkpoint_path: str) -> tuple[int, int]:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint:
            state_dict = checkpoint["ema"].get("module", checkpoint["ema"])
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

    if not isinstance(state_dict, dict):
        return 0, 0

    model_state = model.state_dict()
    compatible = {}
    skipped = 0
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped += 1
            continue
        compatible[key] = value

    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped