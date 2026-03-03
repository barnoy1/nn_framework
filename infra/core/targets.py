from __future__ import annotations

from typing import Dict, List

import torch


def move_targets_to_device(targets: List[Dict], device: torch.device) -> List[Dict]:
    device_targets: List[Dict] = []
    for target in targets:
        packed = {}
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                packed[key] = value.to(device, non_blocking=True)
            else:
                packed[key] = value
        device_targets.append(packed)
    return device_targets
