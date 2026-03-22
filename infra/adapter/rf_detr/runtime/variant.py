from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .config import infer_model_profile, load_dino_config


LOSS_COEFFICIENT_KEYS = (
    "cls_loss_coef",
    "bbox_loss_coef",
    "giou_loss_coef",
    "mask_ce_loss_coef",
    "mask_dice_loss_coef",
)

VARIANT_ALIAS_BY_NORMALIZED_KEY = {
    "xxlarge": "2xlarge",
    "rfdetrxxlarge": "rfdetr2xlarge",
    "rfdetrsegxxlarge": "rfdetrseg2xlarge",
}


def _model_config_classes() -> dict[str, type[Any]]:
    from rfdetr.config import (
        RFDETRBaseConfig,
        RFDETRLargeConfig,
        RFDETRMediumConfig,
        RFDETRNanoConfig,
        RFDETRSeg2XLargeConfig,
        RFDETRSegLargeConfig,
        RFDETRSegMediumConfig,
        RFDETRSegNanoConfig,
        RFDETRSegPreviewConfig,
        RFDETRSegSmallConfig,
        RFDETRSegXLargeConfig,
        RFDETRSmallConfig,
    )

    return {
        "RFDETRBaseConfig": RFDETRBaseConfig,
        "RFDETRNanoConfig": RFDETRNanoConfig,
        "RFDETRSmallConfig": RFDETRSmallConfig,
        "RFDETRMediumConfig": RFDETRMediumConfig,
        "RFDETRLargeConfig": RFDETRLargeConfig,
        "RFDETRSegPreviewConfig": RFDETRSegPreviewConfig,
        "RFDETRSegNanoConfig": RFDETRSegNanoConfig,
        "RFDETRSegSmallConfig": RFDETRSegSmallConfig,
        "RFDETRSegMediumConfig": RFDETRSegMediumConfig,
        "RFDETRSegLargeConfig": RFDETRSegLargeConfig,
        "RFDETRSegXLargeConfig": RFDETRSegXLargeConfig,
        "RFDETRSeg2XLargeConfig": RFDETRSeg2XLargeConfig,
    }


def _is_segmentation_model_class(config_cls: type[Any]) -> bool:
    return "Seg" in config_cls.__name__


def segmentation_enabled(app_config, *, config_name: str = "") -> bool:
    config_cls, _ = _resolve_model_config_class(
        app_config=app_config,
        config_name=config_name,
    )
    return _is_segmentation_model_class(config_cls)


def _resolve_model_variant(config_name: str) -> str:
    from ..schemes import DEFAULT_MODEL_VARIANT, MODEL_VARIANT_TOKENS

    name = str(config_name).lower()
    for token, variant in MODEL_VARIANT_TOKENS:
        if token in name:
            return variant
    return DEFAULT_MODEL_VARIANT


def _normalize_variant_selector(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "")


def _resolve_selected_variant(*, app_config, config_name: str) -> str:
    model_cfg = app_config.model
    if isinstance(model_cfg, dict):
        explicit_variant = str(model_cfg.get("config_variant_name") or "").strip()
    else:
        explicit_variant = str(model_cfg.config_variant_name or "").strip()
    return explicit_variant if explicit_variant else _resolve_model_variant(config_name)


def _resolve_variant_alias(variant: str) -> str:
    return VARIANT_ALIAS_BY_NORMALIZED_KEY.get(variant, variant)


def _read_optional_loss_value(losses_cfg, key: str):
    if isinstance(losses_cfg, dict):
        return losses_cfg.get(key)
    if hasattr(losses_cfg, key):
        return getattr(losses_cfg, key)
    return None


def _build_variant_candidates(*, normalized_variant: str) -> tuple[str, ...]:
    candidates: list[str] = [normalized_variant]

    if normalized_variant.startswith("rfdetrseg"):
        suffix = normalized_variant.replace("rfdetrseg", "", 1)
        candidates.append(f"seg_{suffix}")

    if normalized_variant.startswith("rfdetr"):
        suffix = normalized_variant.replace("rfdetr", "", 1)
        candidates.append(suffix)

    deduped = list(dict.fromkeys(candidates))
    return tuple(deduped)


def _resolve_model_config_class(*, app_config, config_name: str):
    from ..schemes import MODEL_CONFIG_CLASS_BY_VARIANT

    config_classes = _model_config_classes()
    selected_variant = _resolve_selected_variant(
        app_config=app_config,
        config_name=config_name,
    )
    normalized_variant = _resolve_variant_alias(
        _normalize_variant_selector(selected_variant)
    )

    if selected_variant in config_classes:
        return config_classes[selected_variant], normalized_variant

    class_name = None
    candidate_keys = _build_variant_candidates(
        normalized_variant=normalized_variant,
    )
    for candidate in candidate_keys:
        class_name = MODEL_CONFIG_CLASS_BY_VARIANT.get(candidate)
        if class_name is not None:
            break

    if class_name is None:
        class_name = MODEL_CONFIG_CLASS_BY_VARIANT["small"]

    resolved_class = config_classes.get(class_name)
    if resolved_class is None:
        raise ValueError(f"Unsupported RF-DETR config class: {class_name}")

    return resolved_class, normalized_variant


def _build_scheme_overrides(*, variant: str) -> dict[str, Any]:
    from ..schemes import MODEL_CONFIG_OVERRIDES_BY_KEY

    normalized = _resolve_variant_alias(_normalize_variant_selector(variant))
    for candidate in _build_variant_candidates(
        normalized_variant=normalized,
    ):
        override = MODEL_CONFIG_OVERRIDES_BY_KEY.get(candidate)
        if override is not None:
            return deepcopy(override)

    return {}


def build_model_config(*, app_config, config_path: Path):
    config_name = config_path.name
    config_payload = load_dino_config(config_path)
    num_channels = int(config_payload.get("num_channels", 3) or 3)
    model_profile = infer_model_profile(
        config_path=config_path,
        config_payload=config_payload,
    )
    config_cls, variant = _resolve_model_config_class(
        app_config=app_config,
        config_name=config_name,
    )
    use_segmentation = _is_segmentation_model_class(config_cls)
    scheme_overrides = _build_scheme_overrides(variant=variant)

    model_cfg = app_config.model
    if isinstance(model_cfg, dict):
        num_classes = model_cfg["num_classes"]
        num_queries = model_cfg["num_queries"]
        hidden_dim = model_cfg["hidden_dim"]
        losses_cfg = model_cfg["losses"]
    else:
        num_classes = model_cfg.num_classes
        num_queries = model_cfg.num_queries
        hidden_dim = model_cfg.hidden_dim
        losses_cfg = model_cfg.losses

    config_kwargs: dict[str, Any] = {
        "num_classes": num_classes,
        "num_queries": num_queries,
        "num_select": num_queries,
        "hidden_dim": hidden_dim,
        "segmentation_head": use_segmentation,
        "mask_downsample_ratio": 4,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        **model_profile,
    }

    loss_coefficients = {
        "cls_loss_coef": _read_optional_loss_value(losses_cfg, "cls_loss_coef"),
        "bbox_loss_coef": _read_optional_loss_value(losses_cfg, "bbox_loss_coef"),
        "giou_loss_coef": _read_optional_loss_value(losses_cfg, "giou_loss_coef"),
        "mask_ce_loss_coef": _read_optional_loss_value(
            losses_cfg,
            "mask_ce_loss_coef",
        ),
        "mask_dice_loss_coef": _read_optional_loss_value(
            losses_cfg,
            "mask_dice_loss_coef",
        ),
    }
    for key in LOSS_COEFFICIENT_KEYS:
        value = loss_coefficients[key]
        if value is not None:
            config_kwargs[key] = float(value)

    if num_channels != 3:
        config_kwargs["pretrain_weights"] = None

    config_kwargs.update(scheme_overrides)
    return config_cls(**config_kwargs)