from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .adapter import AdapterConfig
from .engine import EngineConfig


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: AdapterConfig
    engine: EngineConfig = Field(default_factory=EngineConfig)

    @field_validator("engine")
    @classmethod
    def validate_paths(cls, value: EngineConfig) -> EngineConfig:
        for dataset_pair in [*value.data.train_sets, *value.data.val_sets]:
            if not dataset_pair.img_dir or not dataset_pair.ann_file:
                raise ValueError("All dataset pairs must define img_dir and ann_file")
        return value

    def ensure_output_dir(self) -> Path:
        output_path = Path(self.engine.execution.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
