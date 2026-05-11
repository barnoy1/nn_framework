from __future__ import annotations

from typing import Any

import torch


def _adapt_first_conv_to_single_channel(*, model_state: dict[str, Any], state_dict: dict[str, Any]) -> None:
    first_conv_key = "enc1.layers.0.weight"
    if first_conv_key not in state_dict or first_conv_key not in model_state:
        return

    checkpoint_weight = state_dict[first_conv_key]
    target_weight = model_state[first_conv_key]
    if not isinstance(checkpoint_weight, torch.Tensor):
        return
    if not isinstance(target_weight, torch.Tensor):
        return
    if checkpoint_weight.ndim != 4 or target_weight.ndim != 4:
        return
    if checkpoint_weight.shape[1] != 3 or target_weight.shape[1] != 1:
        return

    state_dict[first_conv_key] = checkpoint_weight.mean(dim=1, keepdim=True)


def maybe_load_checkpoint(*, runtime_api, model_payload: dict[str, Any]) -> None:
    checkpoint_path = str(model_payload.get("checkpoint_path", "")).strip()
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("tutorial dummy checkpoint must contain a model state_dict mapping")
    model_state = runtime_api.model.state_dict()
    _adapt_first_conv_to_single_channel(model_state=model_state, state_dict=state_dict)
    runtime_api.model.load_state_dict(state_dict, strict=False)
    if hasattr(runtime_api, "metadata") and isinstance(runtime_api.metadata, dict):
        runtime_api.metadata["checkpoint_loaded"] = True
        runtime_api.metadata["checkpoint_channel_adaptation"] = "3ch-to-1ch-first-conv"
