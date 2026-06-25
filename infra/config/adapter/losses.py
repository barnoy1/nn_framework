from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class CriterionLossPairs(BaseModel):
    model_agnostic: list[CriterionLossPair] = Field(default_factory=list)
    model_specific: list[CriterionLossPair] = Field(default_factory=list)

    def iter_model_agnostic(self) -> list[CriterionLossPair]:
        return [*self.model_agnostic]

    def iter_model_specific(self) -> list[CriterionLossPair]:
        return [*self.model_specific]


class ModelLossesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    cls_loss_coef: Optional[float] = None
    bbox_loss_coef: Optional[float] = None
    giou_loss_coef: Optional[float] = None
    dn_cls_loss_coef: Optional[float] = None
    dn_bbox_loss_coef: Optional[float] = None
    fallback_to_model_default: bool = True
    criterion_pairs: CriterionLossPairs = Field(default_factory=CriterionLossPairs)
