from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from .model_builder_base import ReflectiveYamlAdapterModelBuilderBase


@dataclass
class AdapterPipelineState:
    app_config: Any
    repo_root: Path
    config_path: Path
    runtime_config_path: Path
    model: nn.Module | None = None
    criterion: nn.Module | None = None
    postprocessor: nn.Module | None = None
    model_config: Any = None
    model_api: Any = None
    runtime_args: Any = None
    config_payload: dict[str, Any] | None = None
    model_factory: Any = None
    criterion_factory: Any = None
    _scratch: dict[str, Any] = field(default_factory=dict)


STAGE_POSTCONDITIONS: dict[str, tuple[str, ...]] = {
    "head": ("model", "criterion", "postprocessor"),
}


class StagedAdapterModelBuilder(ReflectiveYamlAdapterModelBuilderBase):
    @classmethod
    def manifest(cls):
        raise NotImplementedError("Concrete adapter must expose a manifest()")

    def __init__(self, app_config, repo_root: Path, *, adapter_root: Path) -> None:
        manifest = self.manifest()
        manifest.validate()
        self._manifest = manifest
        self._YAML_CLASS_PATCHES = manifest.yaml_class_patches
        self._RUNTIME_FUNCTION_PATCHES = manifest.runtime_function_patches
        super().__init__(
            app_config=app_config,
            repo_root=repo_root,
            adapter_root=adapter_root,
            config_subdir=manifest.config_subdir,
        )

    def _create_pipeline_state(self) -> AdapterPipelineState:
        config_path = self._resolve_model_config_path()
        runtime_config_path = self._materialize_runtime_compatible_config(config_path)
        return AdapterPipelineState(
            app_config=self.app_config,
            repo_root=self.repo_root,
            config_path=config_path,
            runtime_config_path=runtime_config_path,
        )

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        state = self._create_pipeline_state()
        for stage in self._manifest.override_order:
            overrides = self._manifest.iter_stage_overrides(stage)
            for override in overrides:
                override.apply(builder=self, state=state)
            self._enforce_stage_postconditions(stage, state)

        if self._RUNTIME_FUNCTION_PATCHES:
            self._apply_runtime_patch_manifest(target=state.model)
        return state.model, state.criterion, state.postprocessor

    def _enforce_stage_postconditions(self, stage: str, state: "AdapterPipelineState") -> None:
        for field_name in STAGE_POSTCONDITIONS.get(stage, ()):
            if getattr(state, field_name) is None:
                raise ValueError(
                    f"Adapter {self._manifest.name!r} stage {stage!r} must set "
                    f"{field_name!r} before the stage boundary"
                )


if __name__ == "__main__":
    # ponytail: smallest check that an unset head field fails at the stage
    # boundary; full pipeline is covered by the rf_detr train acceptance gate.
    from types import SimpleNamespace

    builder = StagedAdapterModelBuilder.__new__(StagedAdapterModelBuilder)
    builder._manifest = SimpleNamespace(name="probe")
    state = AdapterPipelineState(
        app_config=None,
        repo_root=Path("."),
        config_path=Path("."),
        runtime_config_path=Path("."),
    )
    state.model = object()
    state.criterion = object()  # postprocessor left None
    try:
        builder._enforce_stage_postconditions("head", state)
        raise AssertionError("head stage must require postprocessor")
    except ValueError as exc:
        assert "postprocessor" in str(exc)
    print("staged builder self-check OK")
