from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    source_root: str
    official_config_path: Optional[str] = None
    model_config_path: Optional[str] = None
    variant: Literal["r18", "r50"]
    num_classes: int
    num_queries: int
    hidden_dim: int
    dn_num_group: int
    sync_bn: bool

    @model_validator(mode="after")
    def normalize_config_path(self) -> "ModelConfig":
        if not self.official_config_path and self.model_config_path:
            self.official_config_path = self.model_config_path
        if not self.model_config_path and self.official_config_path:
            self.model_config_path = self.official_config_path
        if not self.official_config_path:
            raise ValueError("model.official_config_path (or model.model_config_path) must be provided")
        return self
