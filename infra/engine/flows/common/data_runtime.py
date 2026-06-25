from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import ConcatDataset, DataLoader

from infra.config import AppConfig
from infra.data.dataset import COCODetectionDataset, DetectionCollateFn
from infra.data.prep import convert_dataset
from infra.data.transforms import build_albumentations_from_loader


def _extract_dataset_entries(loader_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_cfg = loader_cfg.get("dataset") if isinstance(loader_cfg, dict) else None
    if not isinstance(dataset_cfg, dict):
        return []
    datasets = dataset_cfg.get("datasets")
    if not isinstance(datasets, list):
        return []
    return [entry for entry in datasets if isinstance(entry, dict)]


def _resolve_dataset_transforms(
    *,
    loader_cfg: Dict[str, Any],
    use_masks: bool,
    default_size: int,
    dataset_count: int,
) -> List[Any]:
    global_transforms = build_albumentations_from_loader(
        loader_cfg=loader_cfg,
        use_masks=use_masks,
        default_size=default_size,
    )
    entries = _extract_dataset_entries(loader_cfg)
    if not entries:
        return [global_transforms for _ in range(dataset_count)]

    resolved: List[Any] = []
    for index in range(dataset_count):
        entry = entries[index] if index < len(entries) else {}
        entry_transforms = entry.get("transforms") if isinstance(entry, dict) else None
        if isinstance(entry_transforms, dict):
            scoped_loader_cfg = {"dataset": {"transforms": entry_transforms}}
            transforms = build_albumentations_from_loader(
                loader_cfg=scoped_loader_cfg,
                use_masks=use_masks,
                default_size=default_size,
            )
            resolved.append(transforms)
        else:
            resolved.append(global_transforms)
    return resolved


def prepare_data_if_needed(config: AppConfig) -> None:
    if not config.engine.execution.prepare_data:
        return
    if not config.engine.execution.supervisely_dataset_root:
        raise ValueError(
            "runtime.prepare_data=True requires runtime.supervisely_dataset_root"
        )
    convert_dataset(
        dataset_root=Path(config.engine.execution.supervisely_dataset_root),
        output_dir=Path(config.engine.data.dataset_root),
        splits=config.engine.execution.supervisely_splits,
        ann_subdir=config.engine.execution.ann_subdir,
        img_subdir=config.engine.execution.img_subdir,
    )


def build_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
    use_masks = "segm" in config.engine.data.iou_types
    train_transforms_per_dataset = _resolve_dataset_transforms(
        loader_cfg=config.engine.data.train_dataloader,
        use_masks=use_masks,
        default_size=640,
        dataset_count=len(config.engine.data.train_sets),
    )
    val_transforms_per_dataset = _resolve_dataset_transforms(
        loader_cfg=config.engine.data.val_dataloader,
        use_masks=use_masks,
        default_size=640,
        dataset_count=len(config.engine.data.val_sets),
    )

    train_datasets = [
        COCODetectionDataset(
            img_dir=dataset_pair.img_dir,
            ann_file=dataset_pair.ann_file,
            transforms=train_transforms_per_dataset[index],
            iou_types=config.engine.data.iou_types,
            keep_rle=config.engine.data.keep_rle_in_targets,
            filter_empty_targets=config.engine.data.filter_empty_targets,
            label_mapping=config.engine.data.mapping,
        )
        for index, dataset_pair in enumerate(config.engine.data.train_sets)
    ]
    val_datasets = [
        COCODetectionDataset(
            img_dir=dataset_pair.img_dir,
            ann_file=dataset_pair.ann_file,
            transforms=val_transforms_per_dataset[index],
            iou_types=config.engine.data.iou_types,
            keep_rle=config.engine.data.keep_rle_in_targets,
            filter_empty_targets=False,
            label_mapping=config.engine.data.mapping,
        )
        for index, dataset_pair in enumerate(config.engine.data.val_sets)
    ]

    train_dataset = (
        train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    )
    val_dataset = (
        val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
    )

    collate_fn = DetectionCollateFn()
    worker_count = int(config.engine.train.num_workers)
    train_batch_size = int(config.engine.train.batch_size)
    val_batch_size = int(config.engine.train.val_batch_size)

    loader_worker_kwargs = {}
    if worker_count > 0:
        loader_worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 2,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=worker_count,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=worker_count,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_worker_kwargs,
    )
    return train_loader, val_loader
