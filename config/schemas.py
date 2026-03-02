from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    source_root: str = "raw_models/RT-DETR/rtdetrv2_pytorch"
    official_config_path: str = "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml"
    variant: Literal["r18", "r50"] = "r18"
    num_classes: int = 1
    num_queries: int = 300
    hidden_dim: int = 256
    dn_num_group: int = 5
    sync_bn: bool = True


class TrainConfig(BaseModel):
    epochs: int = 120
    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 4
    lr: float = 2e-4
    backbone_lr_multiplier: float = 1.0
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.1
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    output_dir: str = "./nn_framework/outputs"
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_updates: int = 2000
    seed: int = 42

    @field_validator("backbone_lr_multiplier")
    @classmethod
    def validate_backbone_multiplier(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("backbone_lr_multiplier must be > 0")
        return value


class DataConfig(BaseModel):
    class DatasetPair(BaseModel):
        img_dir: str
        ann_file: str

    dataset_root: Optional[str] = "/workspace/datasets"
    train_sets: List[DatasetPair] = Field(
        default_factory=lambda: [
            DataConfig.DatasetPair(
                img_dir="/workspace/datasets/train/img",
                ann_file="/workspace/datasets/instances_train.json",
            )
        ]
    )
    val_sets: List[DatasetPair] = Field(
        default_factory=lambda: [
            DataConfig.DatasetPair(
                img_dir="/workspace/datasets/valid/img",
                ann_file="/workspace/datasets/instances_valid.json",
            )
        ]
    )

    train_img_dir: Optional[str] = None
    train_ann_file: Optional[str] = None
    val_img_dir: Optional[str] = None
    val_ann_file: Optional[str] = None
    iou_types: List[Literal["bbox", "segm"]] = Field(default_factory=lambda: ["bbox", "segm"])
    filter_empty_targets: bool = True
    keep_rle_in_targets: bool = True

    @field_validator("iou_types")
    @classmethod
    def validate_iou_types(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("iou_types must contain at least one task type")
        return value

    @model_validator(mode="after")
    def normalize_legacy_paths(self) -> "DataConfig":
        if self.train_img_dir and self.train_ann_file:
            self.train_sets = [DataConfig.DatasetPair(img_dir=self.train_img_dir, ann_file=self.train_ann_file)]
        if self.val_img_dir and self.val_ann_file:
            self.val_sets = [DataConfig.DatasetPair(img_dir=self.val_img_dir, ann_file=self.val_ann_file)]

        if not self.train_sets:
            raise ValueError("data.train_sets must contain at least one dataset pair")
        if not self.val_sets:
            raise ValueError("data.val_sets must contain at least one dataset pair")
        return self


class AugConfig(BaseModel):
    image_size: int = 640
    heavy_scale_min: float = 0.5
    heavy_scale_max: float = 1.5
    light_scale_min: float = 0.9
    light_scale_max: float = 1.1
    switch_epoch_ratio: float = 0.85
    horizontal_flip_prob: float = 0.5
    color_jitter_prob: float = 0.3


class RuntimeConfig(BaseModel):
    prepare_data: bool = False
    supervisely_dataset_root: Optional[str] = None
    supervisely_splits: List[str] = Field(default_factory=lambda: ["train", "valid"])
    ann_subdir: str = "ann"
    img_subdir: str = "img"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    aug: AugConfig = Field(default_factory=AugConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("data")
    @classmethod
    def validate_paths(cls, value: DataConfig) -> DataConfig:
        for dataset_pair in [*value.train_sets, *value.val_sets]:
            if not dataset_pair.img_dir or not dataset_pair.ann_file:
                raise ValueError("All dataset pairs must define img_dir and ann_file")
        return value

    def ensure_output_dir(self) -> Path:
        output_path = Path(self.train.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
