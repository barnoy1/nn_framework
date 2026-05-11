from __future__ import annotations

from infra.adapter.core.spec import AdapterManifest

from .overrides import (
    RFDETRConfigOverride,
    RFDETRHeadOverride,
    RFDETRRuntimeOverride,
    RFDETRWeightsOverride,
)
from .schemes import CONFIG_SUBDIR, YAML_CLASS_PATCHES


def create_manifest(*, builder_factory) -> AdapterManifest:
    return AdapterManifest(
        name="rf_detr",
        source_root_aliases=("rf-detr", "rfdetr"),
        builder_factory=builder_factory,
        config_subdir=CONFIG_SUBDIR,
        yaml_class_patches=YAML_CLASS_PATCHES,
        overrides_by_stage={
            "config": (RFDETRConfigOverride(),),
            "runtime": (RFDETRRuntimeOverride(),),
            "weights": (RFDETRWeightsOverride(),),
            "head": (RFDETRHeadOverride(),),
        },
    )
