from __future__ import annotations

REPO_ROOT_TOKEN = "@REPO_ROOT/"
MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"

YAML_CLASS_PATCHES = (
    {
        "module": "src.nn.backbone.presnet",
        "class_name": "PResNet",
        "keys": ("in_channels",),
    },
)

RUNTIME_FUNCTION_PATCHES = ()
