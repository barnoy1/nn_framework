from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
import yaml

from infra.engine.model.ema import EMAModel
from infra.engine.model.losses import (
    CompositeCriterion,
    ConcreteCriterionAdapter,
    DualCriterionSpecResolver,
    prepare_base_criterion_for_agnostic_flow,
)
from infra.engine.model.wrappers.contracts import BuiltComponents, ModelBuilder

from .optimizer_factory import BackboneGroupedAdamWFactory
from .reflection import (
    inject_runtime_functions,
    patch_yaml_class_section,
    patch_yaml_include_tokens,
)


class AgnosticModelBuilderBase(ModelBuilder):
    def __init__(self, app_config, repo_root: Path) -> None:
        self.app_config = app_config
        self.repo_root = repo_root
        self._optimizer_factory = BackboneGroupedAdamWFactory(
            lr=app_config.engine.train.lr,
            weight_decay=app_config.engine.train.weight_decay,
            epochs=app_config.engine.train.epochs,
            backbone_lr_multiplier=app_config.engine.train.backbone_lr_multiplier,
            eta_min_ratio=app_config.engine.train.scheduler.eta_min_ratio,
        )

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        raise NotImplementedError(
            "Concrete model builder must implement build_model_stack"
        )

    def _build_composite_criterion(self, base_criterion, model: nn.Module):
        resolver = DualCriterionSpecResolver.from_app_config(self.app_config)
        prepare_base_criterion_for_agnostic_flow(base_criterion, resolver)
        return CompositeCriterion(
            base_criterion=base_criterion,
            adapters=[ConcreteCriterionAdapter()],
            resolver=resolver,
            dfl_provider=None,
        )

    def build(self) -> BuiltComponents:
        model, base_criterion, postprocessor = self.build_model_stack()
        self.apply_architecture_specifics(
            model=model, targets=[], dn_num_group=self.app_config.adapter.model.dn_num_group
        )
        criterion = self._build_composite_criterion(base_criterion, model=model)
        class_id_to_name = (
            self.app_config.engine.data.class_id_to_name or self.app_config.engine.data.label2classid
        )

        if self.app_config.engine.train.sync_bn and torch.cuda.device_count() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer, scheduler = self._optimizer_factory.build(model)
        ema_model = (
            EMAModel(model, decay=self.app_config.engine.train.ema_decay)
            if self.app_config.engine.train.use_ema
            else None
        )

        return BuiltComponents(
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model,
            class_id_to_name=class_id_to_name,
        )


class ReflectiveYamlAdapterModelBuilderBase(AgnosticModelBuilderBase):
    _REPO_ROOT_TOKEN = "@REPO_ROOT/"
    _MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"
    _YAML_CLASS_PATCHES: tuple[dict[str, Any], ...] = ()
    _RUNTIME_FUNCTION_PATCHES: tuple[dict[str, Any], ...] = ()
    _CONFIG_SUBDIR: tuple[str, ...] = ("configs",)

    def __init__(
        self,
        app_config,
        repo_root: Path,
        *,
        adapter_root: Path,
        config_subdir: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(app_config=app_config, repo_root=repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        self._adapter_root = adapter_root
        self._workspace_root = self.repo_root.parents[2]
        self._config_subdir = config_subdir or self._CONFIG_SUBDIR

    def _resolve_model_config_path(self) -> Path:
        configured_path = str(self.app_config.adapter.model.model_config_path).strip()
        if not configured_path:
            raise ValueError("model.model_config_path must not be empty")

        candidate = Path(configured_path).expanduser()
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            raise FileNotFoundError(f"Model config not found: {candidate}")

        repo_relative = (self.repo_root / candidate).resolve()
        if repo_relative.exists():
            return repo_relative

        adapter_relative = (self._adapter_root / candidate).resolve()
        if adapter_relative.exists():
            return adapter_relative

        adapter_by_name = (
            self._adapter_root.joinpath(*self._config_subdir) / candidate.name
        ).resolve()
        if adapter_by_name.exists():
            return adapter_by_name

        raise FileNotFoundError(
            "Model config not found in model repo or adapter configs: "
            f"configured={configured_path}, repo_candidate={repo_relative}, adapter_candidate={adapter_relative}"
        )

    def _materialize_runtime_compatible_config(self, config_path: Path) -> Path:
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        changed = patch_yaml_include_tokens(
            payload=payload,
            replacements={
                self._REPO_ROOT_TOKEN: str(self._workspace_root),
                self._MODEL_REPO_ROOT_TOKEN: str(self.repo_root),
            },
        )
        changed = self._apply_yaml_patch_manifest(payload=payload) or changed

        if not changed:
            return config_path

        run_configs_dir = self.app_config.ensure_output_dir() / "configs"
        run_configs_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            encoding="utf-8",
            delete=False,
            dir=str(run_configs_dir),
        ) as temp_file:
            yaml.safe_dump(payload, temp_file, sort_keys=False)
            return Path(temp_file.name)

    def _apply_yaml_patch_manifest(self, *, payload: dict[str, Any]) -> bool:
        changed = False
        for patch_spec in self._YAML_CLASS_PATCHES:
            class_name = str(patch_spec["class_name"])
            class_payload = payload.get(class_name)
            keys = tuple(patch_spec.get("keys", ()))
            injected = {
                key: class_payload.get(key) if isinstance(class_payload, dict) else None
                for key in keys
            }
            changed = (
                patch_yaml_class_section(
                    payload=payload,
                    module=str(patch_spec["module"]),
                    class_name=class_name,
                    injected=injected,
                )
                or changed
            )
        return changed

    def _apply_runtime_patch_manifest(self, target: nn.Module) -> None:
        for patch_spec in self._RUNTIME_FUNCTION_PATCHES:
            injected_spec = dict(patch_spec.get("injected", {}))
            injected: dict[str, Callable[..., Any] | None] = {
                str(name): function_impl
                for name, function_impl in injected_spec.items()
            }
            inject_runtime_functions(
                target=target,
                module=str(patch_spec["module"]),
                class_name=str(patch_spec["class_name"]),
                injected=injected,
            )

    def _load_model_config(self):
        raise NotImplementedError("Concrete adapter must implement model-config loader")
