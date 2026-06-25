from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .losses import ModelLossesConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_root: str
    official_config_path: Optional[str] = None
    model_config_path: Optional[str] = None
    num_queries: int
    hidden_dim: int
    dn_num_group: int
    losses: ModelLossesConfig = Field(default_factory=ModelLossesConfig)

    @model_validator(mode="after")
    def normalize_config_path(self) -> "ModelConfig":
        if not self.model_config_path and self.official_config_path:
            self.model_config_path = self.official_config_path
        if not self.official_config_path and self.model_config_path:
            self.official_config_path = self.model_config_path
        if not self.model_config_path:
            raise ValueError(
                "adapter.model.model_config_path (or official_config_path) must be provided"
            )
        return self
