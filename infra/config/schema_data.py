from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetPair(BaseModel):
    img_dir: str
    ann_file: str


class DataConfig(BaseModel):
    dataset_root: Optional[str] = None
    train_sets: List[DatasetPair] = Field(default_factory=list)
    val_sets: List[DatasetPair] = Field(default_factory=list)
    iou_types: List[Literal["bbox", "segm"]] = Field(
        default_factory=lambda: ["bbox", "segm"]
    )
    filter_empty_targets: bool = True
    keep_rle_in_targets: bool = True
    task: str = "detection"
    evaluator: Dict = Field(
        default_factory=lambda: {"type": "CocoEvaluator", "iou_types": ["bbox", "segm"]}
    )
    num_classes: int = 80
    remap_mscoco_category: bool = False
    mapping: Dict[int, int] = Field(default_factory=dict)
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
        def extract_sets(loader_cfg: Dict) -> List[DatasetPair]:
            if not isinstance(loader_cfg, dict):
                return []
            dataset_cfg = loader_cfg.get("dataset")
            if not isinstance(dataset_cfg, dict):
                return []
            datasets_cfg = dataset_cfg.get("datasets")
            if not isinstance(datasets_cfg, list):
                return []

            parsed: List[DatasetPair] = []
            for entry in datasets_cfg:
                if not isinstance(entry, dict):
                    continue
                img_dir = entry.get("img_dir") or entry.get("img_folder")
                ann_file = entry.get("ann_file")
                if img_dir and ann_file:
                    parsed.append(
                        DatasetPair(img_dir=str(img_dir), ann_file=str(ann_file))
                    )
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
            self.class_id_to_name = {
                int(key): str(value) for key, value in self.label2classid.items()
            }
        if not self.label2classid and self.class_id_to_name:
            self.label2classid = {
                int(key): str(value) for key, value in self.class_id_to_name.items()
            }
        return self
