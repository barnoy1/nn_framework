from __future__ import annotations

from pydantic import BaseModel, Field

from .data import DataConfig
from .execution import ExecutionConfig
from .train import TrainConfig


class EngineConfig(BaseModel):
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
