from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from .base import BuiltComponents, CheckpointAdapter, DnGroupConfigurer, ModelBuilder, ModelWrapperAdapter


@dataclass
class FrameworkModelAdapter(ModelWrapperAdapter):
	model_builder: ModelBuilder
	checkpoint_adapter: CheckpointAdapter
	dn_group_configurer: DnGroupConfigurer

	def __post_init__(self) -> None:
		if self.model_builder is None:
			raise ValueError("model_builder must not be None")
		if self.checkpoint_adapter is None:
			raise ValueError("checkpoint_adapter must not be None")
		if self.dn_group_configurer is None:
			raise ValueError("dn_group_configurer must not be None")

	def build_components(self) -> BuiltComponents:
		built = self.model_builder.build()
		if not isinstance(built, BuiltComponents):
			raise TypeError(f"Model builder must return BuiltComponents, got {type(built)!r}")
		return built

	def configure_fixed_dn_num_group(self, model: nn.Module, targets: List[Dict], dn_num_group: int) -> None:
		self.dn_group_configurer(model=model, targets=targets, dn_num_group=dn_num_group)

	def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
		return self.checkpoint_adapter.load_checkpoint_state(path)

	def validate_checkpoint_class_compatibility(
		self,
		model: nn.Module,
		state_dict: Dict[str, torch.Tensor],
	) -> None:
		self.checkpoint_adapter.validate_checkpoint_class_compatibility(
			model=model,
			state_dict=state_dict,
		)

	def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
		return self.checkpoint_adapter.safe_load_state_dict(model=model, state_dict=state_dict)

