from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torchvision

from ..contracts import DFLossProvider


class ModelAgnosticDetCriterion:
    def __init__(
        self,
        *,
        matcher,
        num_classes: int = 80,
        alpha: float = 0.75,
        gamma: float = 2.0,
        box_fmt: str = "cxcywh",
        dfl_provider: Optional[DFLossProvider] = None,
        capability_probe=None,
    ) -> None:
        self.matcher = matcher
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.box_fmt = str(box_fmt)
        self.dfl_provider = dfl_provider
        self.capability_probe = capability_probe
        self._loss_keys = {
            "boxes": ("loss_bbox", "loss_giou"),
            "vfl": ("loss_vfl",),
            "focal": ("loss_focal",),
            "dfl": ("loss_dfl",),
        }

    @classmethod
    def from_base(
        cls,
        base_criterion,
        *,
        dfl_provider: Optional[DFLossProvider] = None,
        capability_probe=None,
    ):
        matcher = getattr(base_criterion, "matcher", None)
        if matcher is None:
            return None
        return cls(
            matcher=matcher,
            num_classes=int(getattr(base_criterion, "num_classes", 80)),
            alpha=float(getattr(base_criterion, "alpha", 0.75)),
            gamma=float(getattr(base_criterion, "gamma", 2.0)),
            box_fmt=str(getattr(base_criterion, "box_fmt", "cxcywh")),
            dfl_provider=dfl_provider or getattr(base_criterion, "dfl_provider", None),
            capability_probe=capability_probe,
        )

    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat(
            [torch.full_like(src, index) for index, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _num_boxes(indices) -> float:
        device = (
            indices[0][0].device
            if indices and len(indices[0]) > 0
            else torch.device("cpu")
        )
        count = sum(len(src) for (src, _) in indices)
        return float(
            torch.clamp(
                torch.as_tensor([count], dtype=torch.float32, device=device), min=1.0
            ).item()
        )

    def _configured(
        self, loss_name: str, resolver, default_weight_dict: Dict[str, float]
    ) -> bool:
        for key in self._loss_keys.get(
            loss_name, ()
        ):  # any positive coef enables the loss
            if float(resolver.resolve(key, default_weight_dict).coef) > 0.0:
                return True
        return False

    def _supported(self, loss_name: str, outputs, targets) -> bool:
        if self.capability_probe is not None and hasattr(
            self.capability_probe, "supports_loss"
        ):
            if not bool(
                self.capability_probe.supports_loss(
                    loss_name, outputs=outputs, targets=targets
                )
            ):
                return False
        if loss_name != "dfl":
            return True
        if self.dfl_provider is not None and hasattr(self.dfl_provider, "supports"):
            return bool(self.dfl_provider.supports(outputs=outputs, targets=targets))
        return self.dfl_provider is not None

    @staticmethod
    def _data_ready(loss_name: str, outputs) -> bool:
        if not isinstance(outputs, dict):
            return False
        if loss_name == "boxes":
            return "pred_boxes" in outputs
        if loss_name == "vfl":
            return "pred_boxes" in outputs and "pred_logits" in outputs
        if loss_name == "focal":
            return "pred_logits" in outputs
        if loss_name == "dfl":
            return True
        return False

    def _enabled_losses(
        self, outputs, targets, *, resolver, default_weight_dict: Dict[str, float]
    ):
        enabled = []
        for name in ("boxes", "vfl", "focal", "dfl"):
            if not self._configured(name, resolver, default_weight_dict):
                continue
            if not self._supported(name, outputs, targets):
                continue
            if not self._data_ready(name, outputs):
                continue
            enabled.append(name)
        return enabled

    def _loss_boxes(
        self, outputs, targets, indices, num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        idx = self._src_idx(indices)
        src = outputs["pred_boxes"][idx]
        tgt = torch.cat(
            [target["boxes"][j] for target, (_, j) in zip(targets, indices)], dim=0
        )
        loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / num_boxes
        src_xyxy = torchvision.ops.box_convert(src, in_fmt=self.box_fmt, out_fmt="xyxy")
        tgt_xyxy = torchvision.ops.box_convert(tgt, in_fmt=self.box_fmt, out_fmt="xyxy")
        loss_giou = (
            1 - torchvision.ops.generalized_box_iou(src_xyxy, tgt_xyxy).diag()
        ).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _loss_vfl(
        self, outputs, targets, indices, num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        idx = self._src_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        tgt_boxes = torch.cat(
            [target["boxes"][j] for target, (_, j) in zip(targets, indices)], dim=0
        )
        src_xyxy = torchvision.ops.box_convert(
            src_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        tgt_xyxy = torchvision.ops.box_convert(
            tgt_boxes, in_fmt=self.box_fmt, out_fmt="xyxy"
        )
        iou = torch.diag(torchvision.ops.box_iou(src_xyxy.detach(), tgt_xyxy))

        src_logits = outputs["pred_logits"]
        tgt_cls_o = torch.cat(
            [target["labels"][j] for target, (_, j) in zip(targets, indices)]
        )
        tgt_cls = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        tgt_cls[idx] = tgt_cls_o
        target = F.one_hot(tgt_cls, num_classes=self.num_classes + 1)[..., :-1]
        tgt_score_o = torch.zeros_like(tgt_cls, dtype=src_logits.dtype)
        tgt_score_o[idx] = iou.to(src_logits.dtype)
        tgt_score = tgt_score_o.unsqueeze(-1) * target

        src_score = F.sigmoid(src_logits.detach())
        weight = self.alpha * src_score.pow(self.gamma) * (1 - target) + tgt_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, tgt_score, weight=weight, reduction="none"
        )
        return {"loss_vfl": loss.sum() / num_boxes}

    def _loss_focal(
        self, outputs, targets, indices, num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]
        idx = self._src_idx(indices)
        tgt_cls_o = torch.cat(
            [target["labels"][j] for target, (_, j) in zip(targets, indices)]
        )
        tgt_cls = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        tgt_cls[idx] = tgt_cls_o
        target = F.one_hot(tgt_cls, num_classes=self.num_classes + 1)[..., :-1].to(
            src_logits.dtype
        )
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        return {"loss_focal": loss.sum() / num_boxes}

    def _loss_dfl(
        self, outputs, targets, indices, num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        if self.dfl_provider is None:
            return {}
        return self.dfl_provider(
            outputs=outputs, targets=targets, indices=indices, num_boxes=num_boxes
        )

    def _apply_weights(
        self,
        loss_dict: Dict[str, torch.Tensor],
        *,
        resolver,
        default_weight_dict: Dict[str, float],
    ):
        weighted: Dict[str, torch.Tensor] = {}
        for key, value in loss_dict.items():
            if value is not None:
                weighted[key] = value * float(
                    resolver.resolve(str(key).strip().lower(), default_weight_dict).coef
                )
        return weighted

    def forward(
        self, outputs, targets, *, resolver, default_weight_dict: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        if (
            self.matcher is None
            or "pred_boxes" not in outputs
            or "pred_logits" not in outputs
        ):
            return {}
        matched = self.matcher(outputs, targets)
        indices = matched.get("indices") if isinstance(matched, dict) else None
        if not indices:
            return {}

        num_boxes = self._num_boxes(indices)
        loss_map = {
            "boxes": self._loss_boxes,
            "vfl": self._loss_vfl,
            "focal": self._loss_focal,
            "dfl": self._loss_dfl,
        }
        raw: Dict[str, torch.Tensor] = {}
        for name in self._enabled_losses(
            outputs, targets, resolver=resolver, default_weight_dict=default_weight_dict
        ):
            raw.update(loss_map[name](outputs, targets, indices, num_boxes))
        return self._apply_weights(
            raw, resolver=resolver, default_weight_dict=default_weight_dict
        )
