## ADDED Requirements

### Requirement: Two-namespace experiment config
The experiment YAML SHALL have exactly two top-level keys: `adapter` (model-owned) and `engine` (framework-owned). `engine` SHALL contain `train`, `data`, and `execution` (renamed from `runtime`). Each setting SHALL have exactly one owner. Loading the old config shape SHALL raise; no deprecation aliases SHALL exist.

#### Scenario: New shape loads clean
- **WHEN** any of the 5 live experiment YAMLs is loaded into `AppConfig`
- **THEN** it parses with zero deprecation aliases and exposes `adapter.*` and `engine.*`

#### Scenario: Old shape rejected
- **WHEN** a config using the pre-refactor top-level keys is loaded
- **THEN** the system raises rather than silently falling back

#### Scenario: Single owner per duplicated setting
- **WHEN** `epochs`, `batch_size`, `num_workers`, or `seed` is read
- **THEN** it resolves to exactly one owner and the old `runtime` duplicates are absent

#### Scenario: rf_detr instance-seg train flow runs end to end
- **WHEN** `python cli.py train --config infra/adapter/rf_detr/experiments/rfdetr_small_coco_instance_seg_rle_1ch.yaml --output-dir <out>` is run after the refactor
- **THEN** the config parses under the new `adapter`/`engine` shape, the adapter resolves by `adapter.name`, the typed build pipeline completes, `segm`-gated masks flow, per-run artifacts land under `<out>/runs/<run_id>/...`, and the process exits 0

### Requirement: Single source of truth for epochs
The scheduler `T_max` and the training loop SHALL both read `engine.train.epochs`. No `runtime.common.epochs` / `train.epochs` split read SHALL remain.

#### Scenario: Scheduler and loop agree
- **WHEN** `engine.train.epochs` is set to N
- **THEN** the scheduler anneals over N epochs and the loop runs N epochs

#### Scenario: No stale split reads
- **WHEN** the codebase is grepped for `runtime.common.epochs` or `train.epochs`
- **THEN** zero matches are found

### Requirement: iou_types is the single task selector
`data.iou_types` SHALL be a subset of `{bbox, segm}` and SHALL be the sole detection-vs-instance-segmentation selector. `data.task` SHALL be deleted. `evaluator.iou_types` SHALL be derived from `data.iou_types`. The presence of `segm` SHALL gate mask load, retain, predict, and metrics.

#### Scenario: Segmentation gated on iou_types
- **WHEN** `iou_types` contains `segm`
- **THEN** masks are loaded, retained, predicted, and scored

#### Scenario: Detection-only omits masks
- **WHEN** `iou_types` is `{bbox}`
- **THEN** no mask pathway is active and `evaluator.iou_types` is `{bbox}`

#### Scenario: task field removed
- **WHEN** a config still containing `data.task` is loaded
- **THEN** the system raises because the field no longer exists

### Requirement: Clean loss vocabulary
Loss groups SHALL be named `model_agnostic` and `model_specific`. Legacy groups `box/cls/dfl/custom`, `default_yolov11_criterion_pairs`, and `normalize_dual_groups` cross-population SHALL be deleted. An empty or partial `model_agnostic` group SHALL be valid; an unsupported model-agnostic loss SHALL be silently inactive (documented contract, not validated).

#### Scenario: Renamed groups resolve
- **WHEN** an adapter declares losses under `model_agnostic` / `model_specific`
- **THEN** the resolver routes them without referencing the legacy group names

#### Scenario: Partial agnostic group valid
- **WHEN** `model_agnostic` is empty or partial
- **THEN** the config is valid and the loop proceeds

#### Scenario: Unsupported agnostic loss inactive
- **WHEN** a model-agnostic loss the model cannot support is declared
- **THEN** it is silently inactive without raising
