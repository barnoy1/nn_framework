from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from infra.config import AppConfig
from infra.data.dataset import COCODetectionDataset, DetectionCollateFn
from infra.data.prep import convert_dataset
from infra.data.transforms import build_albumentations_from_loader


def prepare_data_if_needed(config: AppConfig) -> None:
    if not config.runtime.prepare_data:
        return
    if not config.runtime.supervisely_dataset_root:
        raise ValueError("runtime.prepare_data=True requires runtime.supervisely_dataset_root")
    convert_dataset(
        dataset_root=Path(config.runtime.supervisely_dataset_root),
        output_dir=Path(config.data.dataset_root),
        splits=config.runtime.supervisely_splits,
        ann_subdir=config.runtime.ann_subdir,
        img_subdir=config.runtime.img_subdir,
    )


def build_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
    use_masks = "segm" in config.data.iou_types
    train_transforms = build_albumentations_from_loader(
        loader_cfg=config.data.train_dataloader,
        use_masks=use_masks,
        default_size=640,
    )
    val_transforms = build_albumentations_from_loader(
        loader_cfg=config.data.val_dataloader,
        use_masks=use_masks,
        default_size=640,
    )

    train_datasets = [
        COCODetectionDataset(
            img_dir=dataset_pair.img_dir,
            ann_file=dataset_pair.ann_file,
            transforms=train_transforms,
            iou_types=config.data.iou_types,
            keep_rle=config.data.keep_rle_in_targets,
            filter_empty_targets=config.data.filter_empty_targets,
            label_mapping=config.data.mapping,
        )
        for dataset_pair in config.data.train_sets
    ]
    val_datasets = [
        COCODetectionDataset(
            img_dir=dataset_pair.img_dir,
            ann_file=dataset_pair.ann_file,
            transforms=val_transforms,
            iou_types=config.data.iou_types,
            keep_rle=config.data.keep_rle_in_targets,
            filter_empty_targets=False,
            label_mapping=config.data.mapping,
        )
        for dataset_pair in config.data.val_sets
    ]

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    collate_fn = DetectionCollateFn()
    worker_count = int(config.train.num_workers)
    loader_worker_kwargs = {}
    if worker_count > 0:
        loader_worker_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 2,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=worker_count,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=worker_count,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        **loader_worker_kwargs,
    )
    return train_loader, val_loader
