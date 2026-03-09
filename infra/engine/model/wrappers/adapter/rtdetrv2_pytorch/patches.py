from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision


def patched_loss_labels_focal(self, outputs, targets, indices, num_boxes):
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
