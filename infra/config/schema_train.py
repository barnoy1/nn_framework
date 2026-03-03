from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


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
    output_dir: str = "./nn_framework/outputs"
    log_every_n_steps: int = 20
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_updates: int = 2000
    seed: int = 42

    @field_validator("backbone_lr_multiplier")
    @classmethod
    def validate_backbone_multiplier(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("backbone_lr_multiplier must be > 0")
        return value
