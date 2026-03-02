from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import DynamicAlbumentations


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transforms: Optional[DynamicAlbumentations] = None,
        iou_types: Optional[List[str]] = None,
        keep_rle: bool = True,
        filter_empty_targets: bool = True,
    ) -> None:
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        self.iou_types = iou_types or ["bbox"]
        self.keep_rle = keep_rle
        self.filter_empty_targets = filter_empty_targets

        self.coco = COCO(str(self.ann_file))
        self.image_ids = sorted(self.coco.getImgIds())

        categories = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda cat: cat["id"])
        self.category_id_to_contiguous = {cat["id"]: idx for idx, cat in enumerate(categories)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def _decode_ann_mask(self, annotation: dict) -> np.ndarray:
        decoded = self.coco.annToMask(annotation)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return decoded.astype(np.uint8)

    def _load_annotations(self, image_id: int, width: int, height: int) -> Tuple[List[List[float]], List[int], List[dict]]:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        kept_annotations: List[dict] = []

        for ann in annotations:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            x1 = float(max(0.0, x))
            y1 = float(max(0.0, y))
            x2 = float(min(width, x + w))
            y2 = float(min(height, y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            category_id = ann["category_id"]
            if category_id not in self.category_id_to_contiguous:
                continue
            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(self.category_id_to_contiguous[category_id])
            kept_annotations.append(ann)

        return boxes_xyxy, labels, kept_annotations

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_meta = self.coco.loadImgs([image_id])[0]

        img_path = self.img_dir / image_meta["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        boxes_xyxy, labels, annotations = self._load_annotations(image_id=image_id, width=orig_w, height=orig_h)

        if not boxes_xyxy and self.filter_empty_targets:
            return self[(index + 1) % len(self)]

        masks_np: Optional[List[np.ndarray]] = None
        rle_objects: Optional[List[dict]] = None
        if "segm" in self.iou_types:
            rle_objects = [ann.get("segmentation") for ann in annotations]
            if self.transforms is not None:
                masks_np = [self._decode_ann_mask(ann) for ann in annotations if ann.get("segmentation") is not None]

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes_xyxy, class_labels=labels, masks=masks_np)
            image = transformed.image
            boxes_xyxy = transformed.bboxes
            labels = transformed.class_labels
            if "segm" in self.iou_types and transformed.masks is not None:
                masks_np = [np.asarray(mask, dtype=np.uint8) for mask in transformed.masks]

        if len(boxes_xyxy) == 0:
            boxes_tensor_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            boxes_tensor_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            cx = (boxes_tensor_xyxy[:, 0] + boxes_tensor_xyxy[:, 2]) * 0.5
            cy = (boxes_tensor_xyxy[:, 1] + boxes_tensor_xyxy[:, 3]) * 0.5
            w = boxes_tensor_xyxy[:, 2] - boxes_tensor_xyxy[:, 0]
            h = boxes_tensor_xyxy[:, 3] - boxes_tensor_xyxy[:, 1]
            boxes_tensor_cxcywh = torch.stack(
                [cx / image.shape[1], cy / image.shape[0], w / image.shape[1], h / image.shape[0]], dim=-1
            )

        if boxes_tensor_cxcywh.ndim != 2 or boxes_tensor_cxcywh.shape[-1] != 4:
            raise ValueError("Boxes MUST be [N,4] cxcywh tensor")
        if labels_tensor.ndim != 1:
            raise ValueError("Labels MUST be [N] tensor")

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0

        target: Dict[str, torch.Tensor | List[dict]] = {
            "image_id": torch.tensor([image_id], dtype=torch.long),
            "boxes": boxes_tensor_cxcywh,
            "boxes_xyxy": boxes_tensor_xyxy,
            "labels": labels_tensor,
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.long),
            "size": torch.tensor([image_tensor.shape[2], image_tensor.shape[1]], dtype=torch.long),
        }

        if "segm" in self.iou_types:
            if masks_np is not None:
                masks_tensor = torch.stack([torch.from_numpy(mask).to(torch.uint8) for mask in masks_np], dim=0)
                target["masks"] = masks_tensor
            elif self.keep_rle and rle_objects is not None:
                target["masks_rle"] = rle_objects

        return image_tensor, target


class DetectionCollateFn:
    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = [item[1] for item in batch]
        return images, targets
