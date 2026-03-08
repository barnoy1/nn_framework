from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from .common import GenericCheckpointAdapter
from .contracts import BuiltComponents, ModelBuilder, ModelWrapperAdapter


@dataclass
class FrameworkModelAdapter(ModelWrapperAdapter):
    model_builder: ModelBuilder

    def __post_init__(self) -> None:
        if self.model_builder is None:
            raise ValueError("model_builder must not be None")

        repo_root = getattr(self.model_builder, "repo_root", None)
        if not isinstance(repo_root, Path):
            repo_root = Path.cwd()
        self._checkpoint_adapter = GenericCheckpointAdapter(repo_root=repo_root)

    def build_components(self) -> BuiltComponents:
        built = self.model_builder.build()
        if not isinstance(built, BuiltComponents):
            raise TypeError(f"Model builder must return BuiltComponents, got {type(built)!r}")
        return built

    def apply_architecture_specifics(self, model: nn.Module, targets: List[Dict], *, dn_num_group: int) -> None:
        self.model_builder.apply_architecture_specifics(model=model, targets=targets, dn_num_group=dn_num_group)

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        return self._checkpoint_adapter.load_checkpoint_state(path)

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        self._checkpoint_adapter.validate_checkpoint_class_compatibility(
            model=model,
            state_dict=state_dict,
        )

    def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        return self._checkpoint_adapter.safe_load_state_dict(model=model, state_dict=state_dict)
