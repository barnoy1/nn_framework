from __future__ import annotations

REPO_ROOT_TOKEN = "@REPO_ROOT/"
MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"

CONFIG_SUBDIR = ("dinov2_configs",)

YAML_CLASS_PATCHES = ()

RUNTIME_FUNCTION_PATCHES = ()

DEFAULT_MODEL_VARIANT = "small"

MODEL_CONFIG_CLASS_BY_VARIANT = {
    "base": "RFDETRBaseConfig",
    "nano": "RFDETRNanoConfig",
    "small": "RFDETRSmallConfig",
    "medium": "RFDETRMediumConfig",
    "large": "RFDETRLargeConfig",
    "preview": "RFDETRSegPreviewConfig",
    "seg_nano": "RFDETRSegNanoConfig",
    "seg_small": "RFDETRSegSmallConfig",
    "seg_medium": "RFDETRSegMediumConfig",
    "seg_large": "RFDETRSegLargeConfig",
    "xlarge": "RFDETRSegXLargeConfig",
    "2xlarge": "RFDETRSeg2XLargeConfig",
    "xxlarge": "RFDETRSeg2XLargeConfig",
}

MODEL_VARIANT_TOKENS = (
    ("2xlarge", "2xlarge"),
    ("xxlarge", "xxlarge"),
    ("xlarge", "xlarge"),
    ("preview", "preview"),
    ("medium", "medium"),
    ("small", "small"),
    ("large", "large"),
    ("nano", "nano"),
    ("base", "base"),
)

MODEL_CONFIG_OVERRIDES_BY_KEY = {
    "seg_nano": {
        "dec_layers": 5,
    },
}

SMALL_MODEL_PROFILE = {
    "encoder": "dinov2_windowed_small",
    "num_windows": 2,
    "resolution": 588,
    "out_feature_indexes": [3, 6, 9, 12],
    "projector_scale": ["P4"],
}

BASE_MODEL_PROFILE = {
    "encoder": "dinov2_windowed_base",
    "num_windows": 4,
    "resolution": 560,
    "out_feature_indexes": [2, 5, 8, 11],
    "projector_scale": ["P4"],
}
