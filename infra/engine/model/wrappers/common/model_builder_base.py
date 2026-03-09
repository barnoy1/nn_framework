from __future__ import annotations

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
        self._optimizer_factory = BackboneGroupedAdamWFactory(
            lr=app_config.train.lr,
            weight_decay=app_config.train.weight_decay,
            epochs=app_config.train.epochs,
            backbone_lr_multiplier=app_config.train.backbone_lr_multiplier,
        )

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        raise NotImplementedError("Concrete model builder must implement build_model_stack")

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
        self.apply_architecture_specifics(model=model, targets=[], dn_num_group=self.app_config.model.dn_num_group)
        criterion = self._build_composite_criterion(base_criterion, model=model)
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
