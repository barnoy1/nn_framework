from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["AdamW"] = "AdamW"


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    eta_min_ratio: float = 0.01

    @field_validator("eta_min_ratio")
    @classmethod
    def validate_eta_min_ratio(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("scheduler.eta_min_ratio must be >= 0")
        return value


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
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_updates: int = 2000
    seed: int = 42
    sync_bn: bool = False
    metrics_key: Union[str, list[str]] = "val_loss"
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("engine.train.epochs must be > 0")
        return int(value)

    @field_validator("backbone_lr_multiplier")
    @classmethod
    def validate_backbone_multiplier(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("backbone_lr_multiplier must be > 0")
        return value

    @field_validator("metrics_key")
    @classmethod
    def validate_metrics_key(
        cls, value: Union[str, list[str]]
    ) -> Union[str, list[str]]:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("metrics_key must not be empty")
            return normalized

        normalized_list = [str(item).strip() for item in value if str(item).strip()]
        if not normalized_list:
            raise ValueError("metrics_key must contain at least one metric")
        return normalized_list
