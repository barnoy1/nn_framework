from __future__ import annotations

from typing import Optional

import torch


def expected_model_input_channels(model) -> Optional[int]:
    first_conv_weight = next(
        (
            parameter
            for parameter in model.parameters()
            if getattr(parameter, "ndim", 0) == 4
        ),
        None,
    )
    if first_conv_weight is None:
        return None
    return int(first_conv_weight.shape[1])


def align_images_to_model_input_channels(
    *, images: torch.Tensor, model
) -> torch.Tensor:
    expected_channels = expected_model_input_channels(model)
    if not expected_channels or images.ndim < 4:
        return images

    input_channels = int(images.shape[1])
    if input_channels == expected_channels:
        return images
    if expected_channels == 1:
        return images.mean(dim=1, keepdim=True)
    if input_channels == 1:
        return images.repeat(1, expected_channels, 1, 1)
    return images[:, :expected_channels, :, :]
