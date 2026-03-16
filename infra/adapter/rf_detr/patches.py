from __future__ import annotations

import json
import pickle
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
    configured_variant = str(
        getattr(getattr(app_config, "model", object()), "config_variant_name", "")
        or ""
    ).strip()
    if "seg" in configured_variant.lower():
        return True

    iou_types = list(getattr(app_config.data.evaluator, "iou_types", []) or [])
    if "segm" in iou_types:
        return True
    concrete_specs = (
        getattr(
            getattr(app_config.model.losses, "criterion_pairs", None),
            "concrete_model",
            [],
        )
        or []
    )
    return any(
        "mask" in str(getattr(item, "loss", "")).strip().lower()
        for item in concrete_specs
    )


def _resolve_model_variant(config_name: str) -> str:
    from .schemes import DEFAULT_MODEL_VARIANT, MODEL_VARIANT_TOKENS

    name = str(config_name).lower()
    for token, variant in MODEL_VARIANT_TOKENS:
        if token in name:
            return variant
    return DEFAULT_MODEL_VARIANT


def _normalize_variant_selector(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "")


def _resolve_selected_variant(*, app_config, config_name: str) -> str:
    explicit_variant = str(getattr(app_config.model, "config_variant_name", "") or "").strip()
    if explicit_variant:
        return explicit_variant
    return _resolve_model_variant(config_name)


def _resolve_model_config_class(*, app_config, config_name: str):
    from rfdetr.config import __dict__ as config_symbols

    from .schemes import MODEL_CONFIG_CLASS_BY_VARIANT

    selected_variant = _resolve_selected_variant(
        app_config=app_config,
        config_name=config_name,
    )
    normalized_variant = _normalize_variant_selector(selected_variant)

    if selected_variant in config_symbols:
        return config_symbols[selected_variant], normalized_variant

    class_name = MODEL_CONFIG_CLASS_BY_VARIANT.get(normalized_variant)
    if class_name is None and normalized_variant.startswith("rfdetrseg"):
        suffix = normalized_variant.replace("rfdetrseg", "", 1)
        class_name = MODEL_CONFIG_CLASS_BY_VARIANT.get(f"seg_{suffix}")
    if class_name is None and normalized_variant.startswith("rfdetr"):
        suffix = normalized_variant.replace("rfdetr", "", 1)
        class_name = MODEL_CONFIG_CLASS_BY_VARIANT.get(suffix)
    if class_name is None and normalized_variant in {"nano", "small", "medium", "large"}:
        if segmentation_enabled(app_config):
            class_name = MODEL_CONFIG_CLASS_BY_VARIANT.get(f"seg_{normalized_variant}")
    if class_name is None:
        fallback = "seg_small" if segmentation_enabled(app_config) else "small"
        class_name = MODEL_CONFIG_CLASS_BY_VARIANT[fallback]

    return config_symbols[class_name], normalized_variant


def _build_scheme_overrides(*, variant: str) -> dict[str, Any]:
    from .schemes import MODEL_CONFIG_OVERRIDES_BY_KEY

    normalized = _normalize_variant_selector(variant)
    override = MODEL_CONFIG_OVERRIDES_BY_KEY.get(normalized)
    if override is not None:
        return deepcopy(override)

    if normalized.startswith("rfdetrseg"):
        suffix = normalized.replace("rfdetrseg", "", 1)
        override = MODEL_CONFIG_OVERRIDES_BY_KEY.get(f"seg_{suffix}")
        if override is not None:
            return deepcopy(override)

    if normalized.startswith("rfdetr"):
        suffix = normalized.replace("rfdetr", "", 1)
        override = MODEL_CONFIG_OVERRIDES_BY_KEY.get(suffix)
        if override is not None:
            return deepcopy(override)

    return {}


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
        app_config=app_config,
        config_name=config_name,
    )
    scheme_overrides = _build_scheme_overrides(variant=variant)

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
    for key in (
        "cls_loss_coef",
        "bbox_loss_coef",
        "giou_loss_coef",
        "mask_ce_loss_coef",
        "mask_dice_loss_coef",
    ):
        value = getattr(losses_cfg, key, None)
        if value is not None:
            config_kwargs[key] = float(value)

    disable_pretrain_weights = num_channels != 3
    if disable_pretrain_weights:
        config_kwargs["pretrain_weights"] = None

    config_kwargs.update(scheme_overrides)
    return config_cls(**config_kwargs)


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
