from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Protocol, Tuple

import torch
from torch import nn

from .ema import EMAModel


@dataclass
class BuiltComponents:
    model: nn.Module
    criterion: nn.Module
    postprocessor: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ema_model: EMAModel | None
    class_id_to_name: Dict[int, str] = field(default_factory=dict)


class ModelBuilder(ABC):
    @abstractmethod
    def build(self) -> BuiltComponents:
        raise NotImplementedError


class DnGroupConfigurer(Protocol):
    def __call__(self, model: nn.Module, targets: List[Dict], dn_num_group: int) -> None:
        ...


class CheckpointAdapter(Protocol):
    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        ...

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        ...

    def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        ...


class ModelWrapperAdapter(Protocol):
    def build_components(self) -> BuiltComponents:
        ...

    def configure_fixed_dn_num_group(self, model: nn.Module, targets: List[Dict], dn_num_group: int) -> None:
        ...

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        ...

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        ...

    def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        ...


@dataclass
class WrapperComponents:
    model_builder: ModelBuilder
    checkpoint_adapter: CheckpointAdapter
    dn_group_configurer: DnGroupConfigurer
