from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .schema_data import DataConfig
from .schema_model import ModelConfig
from .schema_runtime import RuntimeConfig
from .schema_train import TrainConfig


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
