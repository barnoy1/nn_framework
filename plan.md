# Implementation Plan: RT-DETRv2 Production Training Architecture

This document outlines the step-by-step execution plan for building a production-grade training stack for RT-DETRv2, supporting both Object Detection and Instance Segmentation (RLE-based).

## 1. Phase 1: Data Preparation & Conversion
**Goal:** Convert Supervisely annotations into RLE-based COCO format.

- [ ] **src/data_prep.py**: Implement the converter.
    - Port logic from `@nn-utilities/helpers/supervisely_to_coco_rle.py`.
    - Ensure `pycocotools.mask` is used for efficient RLE.
    - Map class IDs to contiguous integers starting at 0.
    - Automate conversion for `train` and `valid` splits.
- [ ] **Validation:** Verify `instances_train.json` and `instances_valid.json` exist and contain valid RLE segmentations and bounding boxes.

## 2. Phase 2: Configuration & Schemas
**Goal:** Define a type-safe, hierarchical configuration system.

- [ ] **src/config.py**: Define Pydantic models.
    - `ModelConfig`: Backbone, queries, hidden dims, `dn_num_group`.
    - `TrainConfig`: LRs (with backbone multiplier), weight decay, EMA.
    - `DataConfig`: Paths, batch sizes, `iou_types` (bbox/segm).
    - `AugConfig`: Albumentations parameters.
- [ ] **configs/**: Create Hydra YAML structure.
    - `config.yaml`: Main entry.
    - `model/r18.yaml`, `model/r50.yaml`: Architecture specifics.
    - `experiment/drone_instance_seg.yaml`: Overrides for the drone dataset.

## 3. Phase 3: Dataset & Augmentation
**Goal:** Implement a robust, RLE-aware data pipeline.

- [ ] **src/transforms.py**: Define Albumentations pipelines.
    - Implement `update_augmentation(epoch)` for v2 dynamic intensity.
    - Support both box and mask transformations.
- [ ] **src/dataset.py**: Implement `COCODetectionDataset`.
    - **Critical:** Convert boxes to `cxcywh` normalized FloatTensor.
    - Handle RLE decoding/loading via `pycocotools`.
    - Strict tensor casting and shape assertions (Guard against empty boxes).

## 4. Phase 4: Model Building & EMA
**Goal:** Integrate the official RT-DETRv2 source with EMA and Accelerate.

- [ ] **src/ema.py**: Implement `EMAModel` with `store`/`copy_to`/`restore` methods.
- [ ] **src/model_builder.py**:
    - Import directly from `rtdetrv2_pytorch/src/zoo/rtdetr`.
    - Setup parameter groups (backbone vs encoder/decoder).
    - **Critical:** Register EMA with Accelerator: `accelerator.register_for_checkpointing(ema_model)`.
    - Apply `SyncBatchNorm` before `accelerator.prepare()`.

## 5. Phase 5: Training Loop & Callbacks
**Goal:** Implement the HF Accelerate runner.

- [ ] **src/callbacks.py**: Implement lifecycle hooks.
    - `WandBCallback`: Loss/metric logging + image grids.
    - `CheckpointCallback`: Best/Last saving.
    - `EMACallback`: Swap weights for validation.
    - `DynamicAugCallback`: Trigger augmentation updates.
- [ ] **src/trainer.py**: The core loop.
    - `accelerator.autocast()` for forward + loss.
    - `drop_last=True` for DDP stability.
    - Callback dispatch at each stage.

## 6. Phase 6: Evaluation & Visualization
**Goal:** Metrics and visual debugging.

- [ ] **src/visualize.py**:
    - `rtdetr_output_to_sv_detections()`: Convert raw logits/boxes/masks to `supervision` format.
    - Side-by-side GT vs Pred grid generation.
- [ ] **src/evaluate.py**:
    - Integrate `torchmetrics.detection.MeanAveragePrecision`.
    - Support COCO mask AP via `pycocotools` or `faster-coco-eval`.

## 7. Phase 7: Integration & Entry Point
- [ ] **train.py**: The `@hydra.main` entry point.
    - Orchestrate data prep, config loading, and trainer execution.

---

## Critical Gotchas Memory Map
1. **Denoising Sync:** `dn_num_group` must be a fixed constant in config and passed to both model and criterion.
2. **EMA Resumption:** Must call `register_for_checkpointing` BEFORE `prepare()`.
3. **Coordinate System:** Albumentations list -> Torch stack -> cxcywh normalization [0,1].
4. **EMA Swap:** Always swap on the `unwrapped_model`.
