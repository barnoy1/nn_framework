from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
from torch import nn


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        if device is not None:
            self.ema_model.to(device)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self._shadow_backup: Optional[Dict[str, torch.Tensor]] = None

    def to(self, device: torch.device | str) -> None:
        self.ema_model.to(device)

    def align_to_model(self, model: nn.Module) -> None:
        model_parameter = next(model.parameters(), None)
        if model_parameter is None:
            return
        ema_parameter = next(self.ema_model.parameters(), None)
        if ema_parameter is None:
            return
        if ema_parameter.device != model_parameter.device:
            self.ema_model.to(model_parameter.device)

    def copy_from(self, model: nn.Module) -> None:
        with torch.no_grad():
            self.align_to_model(model)
            self.ema_model.load_state_dict(model.state_dict(), strict=True)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            self.align_to_model(model)
            model_state = model.state_dict()
            for key, ema_value in self.ema_model.state_dict().items():
                model_value = model_state[key].detach().to(device=ema_value.device)
                if not ema_value.dtype.is_floating_point:
                    ema_value.copy_(model_value)
                else:
                    ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        self._shadow_backup = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.ema_model.state_dict(), strict=True)

    def restore(self, model: nn.Module) -> None:
        if self._shadow_backup is None:
            return
        try:
            model.load_state_dict(self._shadow_backup, strict=True)
        except RuntimeError:
            model.load_state_dict(self._shadow_backup, strict=False)
        self._shadow_backup = None

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "ema_model": self.ema_model.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.ema_model.load_state_dict(state_dict["ema_model"], strict=True)
