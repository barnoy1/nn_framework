from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    source_root: str
    official_config_path: str
    variant: Literal["r18", "r50"]
    num_classes: int
    num_queries: int
    hidden_dim: int
    dn_num_group: int
    sync_bn: bool


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

    dataset_root: Optional[str] = None
    train_sets: List[DatasetPair] = Field(default_factory=list)
    val_sets: List[DatasetPair] = Field(default_factory=list)
    iou_types: List[Literal["bbox", "segm"]] = Field(default_factory=lambda: ["bbox", "segm"])
    filter_empty_targets: bool = True
    keep_rle_in_targets: bool = True
    task: str = "detection"
    evaluator: Dict = Field(default_factory=lambda: {"type": "CocoEvaluator", "iou_types": ["bbox", "segm"]})
    num_classes: int = 80
    remap_mscoco_category: bool = False
    label2classid: Dict[int, str] = Field(default_factory=dict)
    class_id_to_name: Dict[int, str] = Field(default_factory=dict)
    train_dataloader: Dict = Field(default_factory=dict)
    val_dataloader: Dict = Field(default_factory=dict)

    @field_validator("iou_types")
    @classmethod
    def validate_iou_types(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("iou_types must contain at least one task type")
        return value

    @model_validator(mode="after")
    def normalize_legacy_paths(self) -> "DataConfig":
        def extract_sets(loader_cfg: Dict) -> List[DataConfig.DatasetPair]:
            if not isinstance(loader_cfg, dict):
                return []
            dataset_cfg = loader_cfg.get("dataset")
            if not isinstance(dataset_cfg, dict):
                return []
            datasets_cfg = dataset_cfg.get("datasets")
            if not isinstance(datasets_cfg, list):
                return []

            parsed: List[DataConfig.DatasetPair] = []
            for entry in datasets_cfg:
                if not isinstance(entry, dict):
                    continue
                img_dir = entry.get("img_dir") or entry.get("img_folder")
                ann_file = entry.get("ann_file")
                if img_dir and ann_file:
                    parsed.append(DataConfig.DatasetPair(img_dir=str(img_dir), ann_file=str(ann_file)))
            return parsed

        if not self.train_sets:
            self.train_sets = extract_sets(self.train_dataloader)
        if not self.val_sets:
            self.val_sets = extract_sets(self.val_dataloader)

        if not self.train_sets:
            raise ValueError("data.train_sets must contain at least one dataset pair")
        if not self.val_sets:
            raise ValueError("data.val_sets must contain at least one dataset pair")

        if not self.class_id_to_name and self.label2classid:
            self.class_id_to_name = {int(key): str(value) for key, value in self.label2classid.items()}
        if not self.label2classid and self.class_id_to_name:
            self.label2classid = {int(key): str(value) for key, value in self.class_id_to_name.items()}
        return self


class RuntimeConfig(BaseModel):
    class ExportConfig(BaseModel):
        post_process: bool = True
        nms: bool = True
        benchmark: bool = False
        fuse_conv_bn: bool = False

    use_gpu: bool = True
    use_xpu: bool = False
    use_mlu: bool = False
    use_npu: bool = False
    log_iter: int = 20
    save_dir: str = "output"
    snapshot_epoch: int = 1
    print_flops: bool = False
    print_params: bool = False
    epoches: Optional[int] = None
    export: ExportConfig = Field(default_factory=ExportConfig)

    prepare_data: bool = False
    supervisely_dataset_root: Optional[str] = None
    supervisely_splits: List[str] = Field(default_factory=lambda: ["train", "valid"])
    ann_subdir: str = "ann"
    img_subdir: str = "img"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @field_validator("epoches")
    @classmethod
    def validate_epoches(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("runtime.epoches must be > 0 when provided")
        return value


class AppConfig(BaseModel):
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
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
