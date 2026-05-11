from __future__ import annotations

from infra.adapter.core.spec import AdapterManifest

from .overrides import (
    DummyConfigOverride,
    DummyHeadOverride,
    DummyRuntimeOverride,
    DummyWeightsOverride,
)
from .schemes import YAML_CLASS_PATCHES


def create_manifest(*, builder_factory) -> AdapterManifest:
    return AdapterManifest(
        name="tutorial_dummy_unet",
        source_root_aliases=("tutorial-dummy-unet", "dummy_unet_tutorial"),
        builder_factory=builder_factory,
        config_subdir=("configs",),
        yaml_class_patches=YAML_CLASS_PATCHES,
        overrides_by_stage={
            "config": (DummyConfigOverride(),),
            "runtime": (DummyRuntimeOverride(),),
            "weights": (DummyWeightsOverride(),),
            "head": (DummyHeadOverride(),),
        },
    )
