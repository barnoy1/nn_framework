from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from infra.engine.model.ema import EMAModel
from infra.engine.model.losses import (
    CompositeCriterion,
    ConcreteCriterionAdapter,
    DualCriterionSpecResolver,
    prepare_base_criterion_for_agnostic_flow,
)
from infra.engine.model.wrappers.contracts import BuiltComponents, ModelBuilder

from .optimizer_factory import BackboneGroupedAdamWFactory


class AgnosticModelBuilderBase(ModelBuilder):
    def __init__(self, app_config, repo_root: Path) -> None:
        self.app_config = app_config
        self.repo_root = repo_root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        self._optimizer_factory = BackboneGroupedAdamWFactory(
            lr=app_config.train.lr,
            weight_decay=app_config.train.weight_decay,
            epochs=app_config.train.epochs,
            backbone_lr_multiplier=app_config.train.backbone_lr_multiplier,
        )

    def _load_model_config(self):
        from src.core import YAMLConfig

        config_rel_path = self.app_config.model.model_config_path or self.app_config.model.model_config_path
        config_path = self.repo_root / str(config_rel_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Official config not found: {config_path}")
        return YAMLConfig(str(config_path))

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
        yaml_cfg = self._load_model_config()
        model = yaml_cfg.model
        self.apply_architecture_specifics(model=model, targets=[], dn_num_group=self.app_config.model.dn_num_group)
        base_criterion = yaml_cfg.criterion
        criterion = self._build_composite_criterion(base_criterion, model=model)
        postprocessor = yaml_cfg.postprocessor
        class_id_to_name = self.app_config.data.class_id_to_name or self.app_config.data.label2classid

        if self.app_config.model.sync_bn and torch.cuda.device_count() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer, scheduler = self._optimizer_factory.build(model)
        ema_model = EMAModel(model, decay=self.app_config.train.ema_decay) if self.app_config.train.use_ema else None

        return BuiltComponents(
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model,
            class_id_to_name=class_id_to_name,
        )
