from __future__ import annotations

from pathlib import Path

from torch import nn

from infra.engine.model.wrappers.common import ReflectiveYamlAdapterModelBuilderBase

from .runtime import (
    apply_local_dinov2_config,
    apply_single_channel_backbone_policy,
    build_model_config,
    ensure_repo_import_paths,
    infer_model_profile,
    load_partial_pretrained_weights,
    load_dino_config,
    maybe_download_pretrain_weights,
    resolve_pretrain_weights_path,
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
        self.model_config = None
        self.model_api = None
        self.args = None
        self.resolution = None
        self._build_criterion_and_postprocessors = None

    def _build_runtime_args(self):
        from rfdetr.main import Model
        from rfdetr.models import build_criterion_and_postprocessors

        self._build_criterion_and_postprocessors = build_criterion_and_postprocessors
        config_path = self._resolve_model_config_path()
        config_payload = load_dino_config(config_path)
        model_profile = infer_model_profile(
            config_path=config_path,
            config_payload=config_payload,
        )
        apply_local_dinov2_config(
            config_path=config_path,
            model_profile=model_profile,
        )
        apply_single_channel_backbone_policy(config_payload=config_payload)
        self.model_config = build_model_config(
            app_config=self.app_config,
            config_path=config_path,
        )
        num_channels = int(config_payload.get("num_channels", 3) or 3)
        if num_channels == 1:
            partial_pretrain_path = resolve_pretrain_weights_path(self.model_config)
            self.model_config.pretrain_weights = None
            self.model_api = Model(**self.model_config.model_dump())
            if partial_pretrain_path:
                load_partial_pretrained_weights(
                    model=self.model_api.model,
                    checkpoint_path=partial_pretrain_path,
                )
        else:
            maybe_download_pretrain_weights(self.model_config)
            self.model_api = Model(**self.model_config.model_dump())
        args = self.model_api.args
        self.args = args
        self.resolution = args.resolution
        return args

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        runtime_args = self._build_runtime_args()
        model = self.model_api.model
        criterion, _ = self._build_criterion_and_postprocessors(runtime_args)
        postprocessor = self.model_api.postprocess
        return model, criterion, postprocessor
