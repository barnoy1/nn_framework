from __future__ import annotations

from pathlib import Path

from torch import nn

from infra.engine.model.wrappers.common import ReflectiveYamlAdapterModelBuilderBase

from .patches import (
    build_runtime_args,
    build_runtime_overrides,
    ensure_repo_import_paths,
    import_entrypoints,
    infer_model_profile,
    load_dino_config,
)

from .schemes import (
    CONFIG_SUBDIR,
    MODEL_REPO_ROOT_TOKEN,
    REPO_ROOT_TOKEN,
    YAML_CLASS_PATCHES,
)


class RFDETRModelBuilder(ReflectiveYamlAdapterModelBuilderBase):
    _REPO_ROOT_TOKEN = REPO_ROOT_TOKEN
    _MODEL_REPO_ROOT_TOKEN = MODEL_REPO_ROOT_TOKEN
    _YAML_CLASS_PATCHES = YAML_CLASS_PATCHES
    _CONFIG_SUBDIR = CONFIG_SUBDIR

    def __init__(self, app_config, repo_root: Path) -> None:
        super().__init__(
            app_config=app_config,
            repo_root=repo_root,
            adapter_root=Path(__file__).resolve().parent,
            config_subdir=self._CONFIG_SUBDIR,
        )
        ensure_repo_import_paths(self.repo_root)
        entrypoints = import_entrypoints()
        self._populate_args = entrypoints["populate_args"]
        self._post_process_cls = entrypoints["post_process_cls"]
        self._build_criterion_and_postprocessors = entrypoints[
            "build_criterion_and_postprocessors"
        ]
        self._build_model = entrypoints["build_model"]
        self.args = None
        self.resolution = None

    def _infer_model_profile(self) -> dict[str, object]:
        config_path = self._resolve_model_config_path()
        config_payload = load_dino_config(config_path)
        return infer_model_profile(config_path=config_path, config_payload=config_payload)

    def _build_runtime_args(self):
        runtime_overrides = build_runtime_overrides(
            app_config=self.app_config,
            model_profile=self._infer_model_profile(),
        )
        args = build_runtime_args(
            populate_args=self._populate_args,
            runtime_overrides=runtime_overrides,
        )
        self.args = args
        self.resolution = args.resolution
        return args

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        runtime_args = self._build_runtime_args()
        model = self._build_model(runtime_args)
        criterion, _ = self._build_criterion_and_postprocessors(runtime_args)
        postprocessor = self._post_process_cls(num_select=runtime_args.num_select)
        return model, criterion, postprocessor