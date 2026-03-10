from __future__ import annotations

REPO_ROOT_TOKEN = "@REPO_ROOT/"
MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"

CONFIG_SUBDIR = ("dinov2_configs",)

YAML_CLASS_PATCHES = ()

RUNTIME_FUNCTION_PATCHES = ()

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
