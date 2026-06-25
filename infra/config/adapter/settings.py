from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .model import ModelConfig


class AdapterConfig(BaseModel):
    name: str
    model: ModelConfig = Field(...)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("adapter.name must not be empty")
        return normalized
