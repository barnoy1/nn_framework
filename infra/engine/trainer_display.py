from __future__ import annotations

from tqdm.auto import tqdm

_HEADER_FORMAT = "%11s" * 7
_ROW_FORMAT = "%11s%11s%11.5f%11.5f%11.5f%11d%11s"


def log_yolo_header(logger) -> None:
    header = _HEADER_FORMAT % (
        "Epoch",
        "GPU_mem",
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "Instances",
        "Size",
    )
    logger.info("{}", header)


def yolo_progress_row(
    *,
    epoch_index: int,
    total_epochs: int,
    gpu_mem_gb: float,
    box_loss: float,
    cls_loss: float,
    dfl_loss: float,
    instances: int,
    image_size: str,
) -> str:
    return _ROW_FORMAT % (
        f"{epoch_index + 1}/{total_epochs}",
        f"{gpu_mem_gb:.3f}G",
        float(box_loss),
        float(cls_loss),
        float(dfl_loss),
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
