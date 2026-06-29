# canonical-prediction

## Purpose

Defines the stable behavior contract for canonical-prediction.

## Requirements

### Requirement: Canonical prediction shape
The postprocessor (head stage) SHALL return a single canonical result shape `{labels, boxes, scores, masks?}` for every model. `masks` SHALL be present if and only if `segm ∈ iou_types`. The 4-branch `to_result_list` shim SHALL be removed.

#### Scenario: Detection result shape
- **WHEN** `iou_types` is `{bbox}` and the model produces predictions
- **THEN** the result contains `labels`, `boxes`, `scores` and no `masks`

#### Scenario: Segmentation result shape
- **WHEN** `iou_types` contains `segm`
- **THEN** the result additionally contains `masks`

#### Scenario: Shim removed
- **WHEN** predictions are post-processed
- **THEN** they pass through the canonical path with no per-model branching shim
