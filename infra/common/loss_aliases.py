from __future__ import annotations


def normalize_loss_name(loss_name: str) -> str:
	return str(loss_name).strip().lower()


def canonical_loss_alias(loss_name: str) -> str:
	normalized = normalize_loss_name(loss_name)
	aliases = {
		"boxes": "loss_bbox",
		"giou": "loss_giou",
		"vfl": "loss_vfl",
		"focal": "loss_focal",
		"loss_boxes": "loss_bbox",
		"loss_boxes_giou": "loss_giou",
		"loss_labels_vfl": "loss_vfl",
		"loss_labels_focal": "loss_focal",
	}
	return aliases.get(normalized, normalized)


__all__ = [
	"canonical_loss_alias",
	"normalize_loss_name",
]
