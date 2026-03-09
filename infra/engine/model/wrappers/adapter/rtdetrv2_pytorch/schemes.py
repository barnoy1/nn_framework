from __future__ import annotations

from .patches import patched_loss_labels_focal

REPO_ROOT_TOKEN = "@REPO_ROOT/"
MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"

YAML_CLASS_PATCHES = (
    {
        "module": "src.nn.backbone.presnet",
        "class_name": "PResNet",
        "keys": ("in_channels",),
    },
)

RUNTIME_FUNCTION_PATCHES = (
    {
        "module": "src.zoo.rtdetr.rtdetrv2_criterion",
        "class_name": "RTDETRCriterionv2",
        "injected": {
            "loss_labels_focal": patched_loss_labels_focal,
        },
    },
)
