from __future__ import annotations

import inspect
import sys
import tempfile
import types
from importlib import import_module
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import yaml

from infra.engine.model.wrappers.common import AgnosticModelBuilderBase


class RTDETRv2ModelBuilder(AgnosticModelBuilderBase):
    _REPO_ROOT_TOKEN = "@REPO_ROOT/"
    _MODEL_REPO_ROOT_TOKEN = "@MODEL_REPO_ROOT/"

    def __init__(self, app_config, repo_root: Path) -> None:
        super().__init__(app_config=app_config, repo_root=repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        self._adapter_root = Path(__file__).resolve().parent
        self._workspace_root = self.repo_root.parents[2]

    def _resolve_model_config_path(self) -> Path:
        configured_path = str(self.app_config.model.model_config_path).strip()
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

        adapter_by_name = (self._adapter_root / "configs" / "rtdetrv2" / candidate.name).resolve()
        if adapter_by_name.exists():
            return adapter_by_name

        raise FileNotFoundError(
            "Model config not found in model repo or adapter configs: "
            f"configured={configured_path}, repo_candidate={repo_relative}, adapter_candidate={adapter_relative}"
        )

    def _load_model_config(self):
        core_module = import_module("src.core")
        yaml_config_cls = getattr(core_module, "YAMLConfig")
        config_path = self._resolve_model_config_path()
        config_path = self._materialize_runtime_compatible_config(config_path)
        return yaml_config_cls(str(config_path))

    def _materialize_runtime_compatible_config(self, config_path: Path) -> Path:
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        changed = False
        includes = payload.get("__include__")
        if isinstance(includes, list):
            resolved_includes: list[str] = []
            for include_path in includes:
                include_text = str(include_path)
                if include_text.startswith(self._REPO_ROOT_TOKEN):
                    relative_to_root = include_text[len(self._REPO_ROOT_TOKEN) :]
                    absolute = (self._workspace_root / relative_to_root).resolve()
                    resolved_includes.append(str(absolute))
                    changed = True
                    continue
                if include_text.startswith(self._MODEL_REPO_ROOT_TOKEN):
                    relative_to_model_root = include_text[len(self._MODEL_REPO_ROOT_TOKEN) :]
                    absolute = (self.repo_root / relative_to_model_root).resolve()
                    resolved_includes.append(str(absolute))
                    changed = True
                    continue
                resolved_includes.append(include_text)
            payload["__include__"] = resolved_includes

        presnet_cfg = payload.get("PResNet")
        if isinstance(presnet_cfg, dict) and "in_channels" in presnet_cfg and not self._presnet_supports_in_channels():
            presnet_cfg = dict(presnet_cfg)
            presnet_cfg.pop("in_channels", None)
            payload["PResNet"] = presnet_cfg
            changed = True

        if not changed:
            return config_path

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            encoding="utf-8",
            delete=False,
            dir=str(config_path.parent),
        ) as temp_file:
            yaml.safe_dump(payload, temp_file, sort_keys=False)
            return Path(temp_file.name)

    @staticmethod
    def _constructor_accepts_parameter(constructor, parameter_name: str) -> bool:
        signature = inspect.signature(constructor)
        if parameter_name in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _presnet_supports_in_channels(self) -> bool:
        module = import_module("src.nn.backbone.presnet")
        constructor = getattr(module, "PResNet").__init__
        return self._constructor_accepts_parameter(constructor, "in_channels")

    @staticmethod
    def _patch_criterion_focal_target_dtype(criterion: nn.Module) -> None:
        has_required_api = all(
            hasattr(criterion, member)
            for member in ("loss_labels_focal", "_get_src_permutation_idx", "num_classes", "alpha", "gamma")
        )
        if not has_required_api:
            return

        def _patched_loss_labels_focal(self, outputs, targets, indices, num_boxes):
            assert "pred_logits" in outputs
            src_logits = outputs["pred_logits"]
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([target["labels"][matched] for target, (_, matched) in zip(targets, indices)])
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(dtype=src_logits.dtype)
            loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction="none")
            loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
            return {"loss_focal": loss}

        criterion.loss_labels_focal = types.MethodType(_patched_loss_labels_focal, criterion)

    def build_model_stack(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        yaml_cfg = self._load_model_config()
        criterion = yaml_cfg.criterion
        self._patch_criterion_focal_target_dtype(criterion)
        return yaml_cfg.model, criterion, yaml_cfg.postprocessor
