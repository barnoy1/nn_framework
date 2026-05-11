from __future__ import annotations

from infra.adapter.core.spec import AdapterManifest

from .overrides import (
    RTDETRv2ConfigOverride,
    RTDETRv2HeadOverride,
    RTDETRv2RuntimeOverride,
    RTDETRv2WeightsOverride,
)
from .schemes import YAML_CLASS_PATCHES


def create_manifest(*, builder_factory) -> AdapterManifest:
    return AdapterManifest(
        name="rtdetrv2_pytorch",
        source_root_tokens=("rtdetrv2_pytorch",),
        builder_factory=builder_factory,
        config_subdir=("configs", "rtdetrv2"),
        yaml_class_patches=YAML_CLASS_PATCHES,
        overrides_by_stage={
            "config": (RTDETRv2ConfigOverride(),),
            "runtime": (RTDETRv2RuntimeOverride(),),
            "weights": (RTDETRv2WeightsOverride(),),
            "head": (RTDETRv2HeadOverride(),),
        },
    )
