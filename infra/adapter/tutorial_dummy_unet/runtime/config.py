from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


def ensure_repo_import_paths(repo_root: Path) -> None:
    for import_path in (
        repo_root,
        repo_root / "src",
        repo_root / "raw_models" / "dummy_unet" / "src",
    ):
        as_text = str(import_path)
        if as_text not in sys.path:
            sys.path.insert(0, as_text)


def load_tutorial_payload(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    model_payload = payload.get("model", payload)
    if not isinstance(model_payload, dict):
        raise ValueError(
            "tutorial dummy adapter expects a mapping at root or under `model`"
        )
    return model_payload


def apply_single_channel_adapter_policy(model_payload: dict[str, Any]) -> dict[str, Any]:
    # Tutorial policy: concrete model may be configured as RGB (3ch), but adapter
    # enforces grayscale runtime (1ch) while preserving original intent for traceability.
    updated = dict(model_payload)
    requested_in_channels = int(updated.get("in_channels", 3))
    updated["requested_in_channels"] = requested_in_channels
    updated["raw_model_in_channels"] = requested_in_channels
    updated["in_channels"] = 1
    updated["effective_model_in_channels"] = 1
    updated["adapter_channel_policy"] = "force-single-channel"
    return updated
