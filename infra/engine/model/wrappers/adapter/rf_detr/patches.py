from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def ensure_repo_import_paths(repo_root: Path) -> None:
	for import_path in (repo_root, repo_root / "src"):
		path_text = str(import_path)
		if path_text not in sys.path:
			sys.path.insert(0, path_text)


def import_entrypoints():
	from rfdetr.main import populate_args
	from rfdetr.models import (
		PostProcess,
		build_criterion_and_postprocessors,
		build_model,
	)

	return {
		"populate_args": populate_args,
		"post_process_cls": PostProcess,
		"build_criterion_and_postprocessors": build_criterion_and_postprocessors,
		"build_model": build_model,
	}


def load_dino_config(config_path: Path) -> dict[str, object]:
	with config_path.open("r", encoding="utf-8") as file:
		return json.load(file)


def infer_model_profile(*, config_path: Path, config_payload: dict[str, object]) -> dict[str, object]:
	from .schemes import BASE_MODEL_PROFILE, SMALL_MODEL_PROFILE

	config_name = str(config_path.name).lower()
	patch_size = int(config_payload.get("patch_size", 14))
	image_size = int(config_payload.get("image_size", 518))

	profile = dict(SMALL_MODEL_PROFILE)
	if "base" in config_name:
		profile = dict(BASE_MODEL_PROFILE)

	profile.update(
		{
			"patch_size": patch_size,
			"positional_encoding_size": image_size // max(1, patch_size),
		}
	)
	return profile


def segmentation_enabled(app_config) -> bool:
	iou_types = list(getattr(app_config.data.evaluator, "iou_types", []) or [])
	return "segm" in iou_types


def build_runtime_overrides(*, app_config, model_profile: dict[str, object]) -> dict[str, object]:
	runtime_overrides = {
		"num_classes": app_config.model.num_classes,
		"num_queries": app_config.model.num_queries,
		"num_select": app_config.model.num_queries,
		"hidden_dim": app_config.model.hidden_dim,
		"segmentation_head": segmentation_enabled(app_config),
		"mask_downsample_ratio": 4,
		"sync_bn": app_config.model.sync_bn,
		"device": ("cuda" if torch.cuda.is_available() else "cpu"),
		**model_profile,
	}

	losses_cfg = app_config.model.losses
	for key in ("cls_loss_coef", "bbox_loss_coef", "giou_loss_coef"):
		value = getattr(losses_cfg, key, None)
		if value is not None:
			runtime_overrides[key] = float(value)
	return runtime_overrides


def build_runtime_args(*, populate_args, runtime_overrides: dict[str, object]) -> argparse.Namespace:
	return populate_args(**runtime_overrides)

