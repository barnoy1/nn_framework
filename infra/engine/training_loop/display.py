from __future__ import annotations

from typing import Dict

from tqdm.auto import tqdm

_COL_WIDTH = 11


def _format_loss_value(value: float | None) -> str:
    if value is None:
        return "0.00000"
    return f"{float(value):.5f}"


def _render_row(columns: list[object]) -> str:
    return " ".join(f"{str(value):>{_COL_WIDTH}}" for value in columns)


def _display_bucket_name(bucket_name: str) -> str:
    normalized = str(bucket_name).strip()
    if normalized.startswith("common_"):
        return normalized[len("common_") :]
    return normalized


def log_yolo_header(
    logger,
    *,
    common_bucket_names: list[str],
) -> None:
    display_bucket_names = [_display_bucket_name(name) for name in common_bucket_names]
    header_columns = [
        "Epoch",
        "GPU_mem",
        *display_bucket_names,
        "custom_loss",
        "Instances",
        "Size",
    ]
    header = _render_row(header_columns)
    logger.info("\n{}", header)


def yolo_progress_row(
    *,
    epoch_index: int,
    total_epochs: int,
    gpu_mem_gb: float,
    custom_loss: float,
    common_bucket_names: list[str],
    common_bucket_values: Dict[str, float],
    instances: int,
    image_size: str,
) -> str:
    common_values = [
        _format_loss_value(common_bucket_values.get(name, 0.0))
        for name in common_bucket_names
    ]
    return _render_row(
        [
            f"{epoch_index + 1}/{total_epochs}",
            f"{gpu_mem_gb:.3f}G",
            _format_loss_value(custom_loss),
            *common_values,
            int(instances),
            str(image_size),
        ]
    )


def create_epoch_progress_bar(train_loader):
    return tqdm(
        train_loader,
        total=len(train_loader),
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
    )
