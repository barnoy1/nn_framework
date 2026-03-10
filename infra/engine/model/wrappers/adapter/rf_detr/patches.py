from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def ensure_repo_import_paths(repo_root: Path) -> None:
    for import_path in (repo_root, repo_root / "src"):
        path_text = str(import_path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def load_dino_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_model_profile(
    *, config_path: Path, config_payload: dict[str, object]
) -> dict[str, object]:
    from .schemes import BASE_MODEL_PROFILE, SMALL_MODEL_PROFILE

    config_name = str(config_path.name).lower()
    patch_size = int(config_payload.get("patch_size", 14))
    image_size = int(config_payload.get("image_size", 518))

    profile = dict(SMALL_MODEL_PROFILE)
    if "base" in config_name:
        profile = dict(BASE_MODEL_PROFILE)

    profile.update(
        {
            "patch_size": patch_size,
            "resolution": image_size,
            "positional_encoding_size": image_size // max(1, patch_size),
        }
    )
    return profile


def apply_local_dinov2_config(
    *, config_path: Path, model_profile: dict[str, object]
) -> None:
    import rfdetr.models.backbone.dinov2 as dinov2_module

    encoder_name = str(model_profile.get("encoder", "dinov2_windowed_small")).lower()
    size = "base" if "base" in encoder_name else "small"
    resolved_config_path = str(config_path.resolve())

    dinov2_module.size_to_config[size] = resolved_config_path
    dinov2_module.size_to_config_with_registers[size] = resolved_config_path


def apply_single_channel_backbone_policy(*, config_payload: dict[str, object]) -> None:
    from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        WindowedDinov2WithRegistersBackbone,
    )

    num_channels = int(config_payload.get("num_channels", 3) or 3)
    if num_channels == 3:
        return
    if getattr(
        WindowedDinov2WithRegistersBackbone, "_nnf_ignore_mismatch_patch", False
    ):
        return

    original_from_pretrained = (
        WindowedDinov2WithRegistersBackbone.from_pretrained.__func__
    )

    def _patched_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("ignore_mismatched_sizes", True)
        return original_from_pretrained(cls, *args, **kwargs)

    WindowedDinov2WithRegistersBackbone.from_pretrained = classmethod(
        _patched_from_pretrained
    )
    WindowedDinov2WithRegistersBackbone._nnf_ignore_mismatch_patch = True


def segmentation_enabled(app_config) -> bool:
    iou_types = list(getattr(app_config.data.evaluator, "iou_types", []) or [])
    return "segm" in iou_types


def _resolve_model_variant(config_name: str) -> str:
    from .schemes import DEFAULT_MODEL_VARIANT, MODEL_VARIANT_TOKENS

    name = str(config_name).lower()
    for token, variant in MODEL_VARIANT_TOKENS:
        if token in name:
            return variant
    return DEFAULT_MODEL_VARIANT


def _resolve_model_config_class(*, config_name: str, use_segmentation: bool):
    from rfdetr.config import __dict__ as config_symbols

    from .schemes import (
        DETECTION_MODEL_CONFIG_CLASS_BY_VARIANT,
        SEGMENTATION_MODEL_CONFIG_CLASS_BY_VARIANT,
    )

    variant = _resolve_model_variant(config_name)
    mapping = (
        SEGMENTATION_MODEL_CONFIG_CLASS_BY_VARIANT
        if use_segmentation
        else DETECTION_MODEL_CONFIG_CLASS_BY_VARIANT
    )
    class_name = mapping.get(variant)
    if class_name is None:
        fallback = "small" if not use_segmentation else "small"
        class_name = mapping[fallback]
    return config_symbols[class_name], variant


def _build_scheme_overrides(*, variant: str, use_segmentation: bool) -> dict[str, Any]:
    from .schemes import MODEL_CONFIG_OVERRIDES_BY_KEY

    key = f"seg_{variant}" if use_segmentation else variant
    return deepcopy(MODEL_CONFIG_OVERRIDES_BY_KEY.get(key, {}))


def build_model_config(*, app_config, config_path: Path):
    config_name = config_path.name
    config_payload = load_dino_config(config_path)
    num_channels = int(config_payload.get("num_channels", 3) or 3)
    patch_size = int(config_payload.get("patch_size", 16) or 16)
    model_profile = infer_model_profile(
        config_path=config_path,
        config_payload=config_payload,
    )
    use_segmentation = segmentation_enabled(app_config)
    config_cls, variant = _resolve_model_config_class(
        config_name=config_name,
        use_segmentation=use_segmentation,
    )
    scheme_overrides = _build_scheme_overrides(
        variant=variant,
        use_segmentation=use_segmentation,
    )

    config_kwargs: dict[str, Any] = {
        "num_classes": app_config.model.num_classes,
        "num_queries": app_config.model.num_queries,
        "num_select": app_config.model.num_queries,
        "hidden_dim": app_config.model.hidden_dim,
        "segmentation_head": use_segmentation,
        "mask_downsample_ratio": 4,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        **model_profile,
    }

    losses_cfg = app_config.model.losses
    for key in ("cls_loss_coef", "bbox_loss_coef", "giou_loss_coef"):
        value = getattr(losses_cfg, key, None)
        if value is not None:
            config_kwargs[key] = float(value)

    is_non_default_pretrain_geometry = num_channels != 3 or patch_size != 16
    if is_non_default_pretrain_geometry:
        config_kwargs["pretrain_weights"] = None

    config_kwargs.update(scheme_overrides)
    return config_cls(**config_kwargs)


def maybe_download_pretrain_weights(model_config) -> None:
    from rfdetr.assets.model_weights import download_pretrain_weights

    pretrain_weights = getattr(model_config, "pretrain_weights", None)
    if pretrain_weights is None:
        return

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

    model_config.pretrain_weights = str(resolved_path.resolve())
