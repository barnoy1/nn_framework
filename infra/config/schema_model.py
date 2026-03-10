from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CriterionLossPair(BaseModel):
    loss: str
    coef: Optional[float] = None

    @field_validator("coef")
    @classmethod
    def validate_coef(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("loss coefficient must be finite")
        if numeric < 0.0:
            raise ValueError("loss coefficient must be non-negative")
        return numeric


def default_yolov11_criterion_pairs() -> "CriterionLossPairs":
    return CriterionLossPairs(
        box=[CriterionLossPair(loss="loss_bbox", coef=7.5)],
        cls=[CriterionLossPair(loss="loss_cls", coef=0.5)],
        dfl=[CriterionLossPair(loss="loss_dfl", coef=1.5)],
    )


class CriterionLossPairs(BaseModel):
    # Legacy groups (kept for backward compatibility).
    box: list[CriterionLossPair] = Field(default_factory=list)
    cls: list[CriterionLossPair] = Field(default_factory=list)
    dfl: list[CriterionLossPair] = Field(default_factory=list)
    custom: list[CriterionLossPair] = Field(default_factory=list)

    # New dual-criterion groups.
    adapter_common: list[CriterionLossPair] = Field(default_factory=list)
    concrete_model: list[CriterionLossPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_dual_groups(self) -> "CriterionLossPairs":
        if len(self.adapter_common) == 0:
            self.adapter_common = [*self.box, *self.cls, *self.dfl]

        if len(self.concrete_model) == 0:
            self.concrete_model = [*self.custom]

        # Keep legacy groups available for existing framework components.
        if len(self.box) == 0 and len(self.cls) == 0 and len(self.dfl) == 0:
            for item in self.adapter_common:
                normalized = str(item.loss).strip().lower()
                if "dfl" in normalized:
                    self.dfl.append(item)
                elif any(token in normalized for token in ("bbox", "giou", "boxes")):
                    self.box.append(item)
                elif any(
                    token in normalized for token in ("vfl", "focal", "cls", "label")
                ):
                    self.cls.append(item)

        if len(self.custom) == 0 and len(self.concrete_model) > 0:
            self.custom = [*self.concrete_model]

        if len(self.adapter_common) == 0 and len(self.concrete_model) == 0:
            raise ValueError(
                "model.losses.criterion_pairs must define losses in legacy groups "
                "(box/cls/dfl/custom) or new groups (adapter_common/concrete_model)"
            )

        return self

    def iter_adapter_common(self) -> list[CriterionLossPair]:
        return [*self.adapter_common]

    def iter_concrete_model(self) -> list[CriterionLossPair]:
        return [*self.concrete_model]


class ModelLossesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    cls_loss_coef: Optional[float] = None
    bbox_loss_coef: Optional[float] = None
    giou_loss_coef: Optional[float] = None
    dn_cls_loss_coef: Optional[float] = None
    dn_bbox_loss_coef: Optional[float] = None
    fallback_to_model_default: bool = True
    criterion_pairs: CriterionLossPairs = Field(
        default_factory=default_yolov11_criterion_pairs
    )


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
            raise ValueError(
                "model.model_config_path (or model.official_config_path) must be provided"
            )
        return self
