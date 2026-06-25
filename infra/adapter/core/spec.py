from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from infra.engine.model.wrappers.contracts import ModelBuilder


BuilderFactory = Callable[[Any, Path], ModelBuilder]
AdapterStage = Literal["config", "runtime", "weights", "head"]
DEFAULT_OVERRIDE_ORDER: tuple[AdapterStage, ...] = (
    "config",
    "runtime",
    "weights",
    "head",
)


@dataclass(frozen=True)
class AdapterOverride(Protocol):
    def apply(self, *, builder: ModelBuilder, state: Any) -> None:
        ...


@dataclass(frozen=True)
class AdapterManifest:
    name: str
    builder_factory: BuilderFactory
    overrides_by_stage: Mapping[AdapterStage, tuple[AdapterOverride, ...]]
    override_order: tuple[AdapterStage, ...] = DEFAULT_OVERRIDE_ORDER
    config_subdir: tuple[str, ...] = ("configs",)
    yaml_class_patches: tuple[dict[str, Any], ...] = ()
    runtime_function_patches: tuple[dict[str, Any], ...] = ()

    def iter_stage_overrides(self, stage: AdapterStage) -> tuple[AdapterOverride, ...]:
        return tuple(self.overrides_by_stage.get(stage, ()))

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Adapter manifest name must not be empty")
        if self.override_order != DEFAULT_OVERRIDE_ORDER:
            raise ValueError(
                f"Adapter manifest {self.name!r} must use deterministic override order "
                f"{DEFAULT_OVERRIDE_ORDER}"
            )
        missing = [
            stage
            for stage in DEFAULT_OVERRIDE_ORDER
            if stage not in self.overrides_by_stage
        ]
        if missing:
            raise ValueError(
                f"Adapter manifest {self.name!r} missing override stages: {missing}"
            )
        multi = [
            stage
            for stage, overrides in self.overrides_by_stage.items()
            if len(overrides) > 1
        ]
        if multi:
            raise ValueError(
                f"Adapter manifest {self.name!r} must define at most one override "
                f"per stage; offending stages: {multi}"
            )


# Backward-compatible export name used by existing imports.
AdapterSpec = AdapterManifest
