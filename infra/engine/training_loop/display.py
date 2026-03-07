from __future__ import annotations

from typing import Optional

from tqdm.auto import tqdm

_HEADER_FORMAT = "%11s" * 7
_ROW_FORMAT = "%11s%11s%11s%11s%11s%11d%11s"


def _format_loss_value(value: Optional[float], enabled: bool) -> str:
    if not enabled or value is None:
        return "N/A"
    return f"{float(value):.5f}"


def log_yolo_header(logger, *, dfl_enabled: bool = True) -> None:
    header = _HEADER_FORMAT % (
        "Epoch",
        "GPU_mem",
        "box_loss",
        "cls_loss",
        "dfl_loss" if dfl_enabled else "dfl(N/A)",
        "Instances",
        "Size",
    )
    logger.info("\n{}", header)


def yolo_progress_row(
    *,
    epoch_index: int,
    total_epochs: int,
    gpu_mem_gb: float,
    box_loss: float,
    cls_loss: float,
    dfl_loss: Optional[float],
    dfl_enabled: bool,
    instances: int,
    image_size: str,
) -> str:
    return _ROW_FORMAT % (
        f"{epoch_index + 1}/{total_epochs}",
        f"{gpu_mem_gb:.3f}G",
        _format_loss_value(box_loss, True),
        _format_loss_value(cls_loss, True),
        _format_loss_value(dfl_loss, dfl_enabled),
        int(instances),
        str(image_size),
    )


def create_epoch_progress_bar(train_loader):
    return tqdm(
        train_loader,
        total=len(train_loader),
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar:20}{r_bar}",
    )
