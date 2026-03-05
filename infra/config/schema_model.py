from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CriterionLossPair(BaseModel):
    loss: str
    coef: Optional[float] = None


class CriterionLossPairs(BaseModel):
    box: list[CriterionLossPair] = Field(default_factory=list)
    cls: list[CriterionLossPair] = Field(default_factory=list)
    dfl: list[CriterionLossPair] = Field(default_factory=list)
    custom: list[CriterionLossPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_groups(self) -> "CriterionLossPairs":
        if len(self.box) == 0:
            raise ValueError("model.losses.criterion_pairs.box must include at least one loss")
        if len(self.cls) == 0:
            raise ValueError("model.losses.criterion_pairs.cls must include at least one loss")
        return self


class ModelLossesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    cls_loss_coef: Optional[float] = None
    bbox_loss_coef: Optional[float] = None
    giou_loss_coef: Optional[float] = None
    dn_cls_loss_coef: Optional[float] = None
    dn_bbox_loss_coef: Optional[float] = None
    criterion_pairs: CriterionLossPairs = Field(default_factory=CriterionLossPairs)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_root: str
    official_config_path: Optional[str] = None
    model_config_path: Optional[str] = None
    num_classes: int
    num_queries: int
    hidden_dim: int
    dn_num_group: int
    sync_bn: bool
    losses: ModelLossesConfig = Field(default_factory=ModelLossesConfig)

    @model_validator(mode="after")
    def normalize_config_path(self) -> "ModelConfig":
        if not self.model_config_path and self.official_config_path:
            self.model_config_path = self.official_config_path
        if not self.official_config_path and self.model_config_path:
            self.official_config_path = self.model_config_path
        if not self.model_config_path:
            raise ValueError("model.model_config_path (or model.official_config_path) must be provided")
        return self
