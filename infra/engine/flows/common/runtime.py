from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

INFRA_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = INFRA_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.config import AppConfig
from infra.data.dataset import COCODetectionDataset, DetectionCollateFn
from infra.data.prep import convert_dataset
from infra.data.transforms import build_albumentations_from_loader
from infra.engine.model import (
    BuiltComponents,
    ModelWrapperAdapter,
    create_model_wrapper,
)

@dataclass(frozen=True)
class FlowRuntime:
    app_config: AppConfig
    built: BuiltComponents
    wrapper: ModelWrapperAdapter
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]


def load_app_config(model_profile: str, overrides: List[str], config_path: str) -> AppConfig:
    config_dir = INFRA_ROOT / "config" / "hydra"

    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        relative = config_file.relative_to(config_dir)
    except ValueError as error:
        raise ValueError(f"Config file must be under {config_dir}: {config_file}") from error

    config_name = str(relative.with_suffix("")).replace("\\", "/")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    payload = OmegaConf.to_container(cfg, resolve=True)
    return AppConfig.model_validate(payload)


def resolve_model_root(config: AppConfig) -> Path:
    candidate = Path(config.model.source_root).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _prepare_data_if_needed(config: AppConfig) -> None:
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


def _build_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_flow_runtime(model_profile: str, overrides: List[str], config_path: str, build_loaders: bool = True) -> FlowRuntime:
    config = load_app_config(model_profile=model_profile, overrides=overrides, config_path=config_path)
    if build_loaders:
        config.ensure_output_dir()
    _prepare_data_if_needed(config)

    if build_loaders:
        train_loader, val_loader = _build_loaders(config)
    else:
        train_loader, val_loader = None, None

    model_root = resolve_model_root(config)
    wrapper = create_model_wrapper(config, repo_root=model_root)
    built = wrapper.build_components()

    return FlowRuntime(
        app_config=config,
        built=built,
        wrapper=wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
    )
