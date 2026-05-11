from __future__ import annotations

from typing import Any


def build_runtime_api(model_payload: dict[str, Any]):
    from raw_models.dummy_unet.src.api import DummyUNetRuntimeAPI

    return DummyUNetRuntimeAPI.from_payload(model_payload)
