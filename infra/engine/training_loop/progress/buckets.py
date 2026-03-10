from __future__ import annotations

from typing import Mapping

from infra.common.loss_aliases import canonical_loss_alias
from ..display import yolo_progress_row


_DISPLAY_BUCKET_ORDER = (
    "common_loss_bbox",
    "common_loss_giou",
    "common_loss_vfl",
    "common_loss_focal",
)


def build_common_bucket_names(trainer) -> list[str]:
    configured = list(
        dict.fromkeys(
            [
                f"common_{canonical_loss_alias(str(item.loss)).rstrip('_')}"
                for item in trainer.app_config.model.losses.criterion_pairs.iter_adapter_common()
            ]
        )
    )
    configured_set = set(configured)
    ordered_display = [
        bucket_name
        for bucket_name in _DISPLAY_BUCKET_ORDER
        if bucket_name in configured_set
    ]
    if ordered_display:
        return ordered_display
    return configured


def collect_bucket_values(
    source: Mapping[str, object], bucket_names: list[str]
) -> dict[str, float]:
    return {name: float(source.get(name, 0.0)) for name in bucket_names}


def build_progress_row(
    *,
    epoch_index: int,
    total_epochs: int,
    gpu_mem_gb: float,
    values: Mapping[str, object],
    common_bucket_names: list[str],
    instances: int,
    image_size: str,
) -> str:
    return yolo_progress_row(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        gpu_mem_gb=gpu_mem_gb,
        custom_loss=float(values.get("custom_loss", 0.0)),
        common_bucket_names=common_bucket_names,
        common_bucket_values=collect_bucket_values(values, common_bucket_names),
        instances=instances,
        image_size=image_size,
    )
