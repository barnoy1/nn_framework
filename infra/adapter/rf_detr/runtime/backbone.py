from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


_SINGLE_CHANNEL_POLICY_PATCHED = False


def apply_local_dinov2_config(
    *, config_path: Path, model_profile: dict[str, object]
) -> None:
    import rfdetr.models.backbone.dinov2 as dinov2_module

    encoder_name = str(model_profile.get("encoder", "dinov2_windowed_small")).lower()
    size = "base" if "base" in encoder_name else "small"
    resolved_config_path = str(config_path.resolve())
    dinov2_module.size_to_config[size] = resolved_config_path
    dinov2_module.size_to_config_with_registers[size] = resolved_config_path


def apply_single_channel_backbone_policy(
    *, config_payload: DictConfig
) -> None:
    from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        WindowedDinov2WithRegistersBackbone,
    )

    global _SINGLE_CHANNEL_POLICY_PATCHED

    if not isinstance(config_payload, DictConfig):
        raise TypeError("RF-DETR backbone policy expects OmegaConf DictConfig")
    num_channels = int(
        config_payload.num_channels if "num_channels" in config_payload else 3
    )
    if num_channels == 3:
        return
    if _SINGLE_CHANNEL_POLICY_PATCHED:
        return

    original_from_pretrained = WindowedDinov2WithRegistersBackbone.from_pretrained

    def _patched_from_pretrained(cls, *args, **kwargs):
        patched_kwargs = dict(kwargs)
        if "ignore_mismatched_sizes" not in patched_kwargs:
            patched_kwargs["ignore_mismatched_sizes"] = True
        return original_from_pretrained(*args, **patched_kwargs)

    WindowedDinov2WithRegistersBackbone.from_pretrained = classmethod(
        _patched_from_pretrained
    )
    _SINGLE_CHANNEL_POLICY_PATCHED = True