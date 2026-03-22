from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from infra.engine.model.wrappers.contracts import ModelBuilder


BuilderFactory = Callable[[Any, Path], ModelBuilder]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    source_root_tokens: tuple[str, ...]
    builder_factory: BuilderFactory

    def matches_source_root(self, source_root: str) -> bool:
        normalized = str(source_root).strip().lower()
        if not normalized:
            return False
        return any(token in normalized for token in self.source_root_tokens)