## Why

`nn_framework` was reverse-engineered from YOLO and later grew an adapter layer, leaving a config and build surface with overlapping ownership: "adapter" is overloaded, the framework/adapter boundary is invisible in config, task selection and loss groups carry dead/duplicated fields, and several values have two sources of truth — one of which silently corrupts training (LR anneals over 120 epochs while the loop runs 250). This refactor makes ownership explicit and fail-fast before more models are onboarded onto a foundation that mis-routes silently.

## What Changes

- **BREAKING** Reorganize experiment YAML into exactly two top-level namespaces: `adapter` (model-owned) and `engine` (framework-owned), with one owner per setting. No deprecation aliases — loading the old shape raises.
- **BREAKING** Select adapters by explicit `adapter.name`; remove `matches_source_root` substring matching. Unknown / zero / multiple matches → hard error.
- **BREAKING** `data.iou_types ⊆ {bbox, segm}` becomes the sole task selector; delete `data.task`; derive `evaluator.iou_types` from it.
- **BREAKING** Rename loss groups `adapter_common→model_agnostic`, `concrete_model→model_specific`; delete legacy `box/cls/dfl/custom`, `default_yolov11_criterion_pairs`, and `normalize_dual_groups`. Unsupported agnostic losses are silently inactive (documented contract).
- Make the postprocessor emit a canonical `{labels, boxes, scores, masks?}` (masks iff `segm ∈ iou_types`); remove the 4-branch `to_result_list` shim.
- Flow-gate checkpoint strictness: train = permissive + loud summary; eval/inference = strict fail-fast with `--allow-partial` opt-out; head class-count mismatch in eval = always hard error.
- Per-run artifact dirs under `<output_base>/runs/<run_id>/...` with shared `mlflow` store and `tensorboard/<run_id>`; rename "visualization logger" → `ExperimentTracker`. MLflow stays default-on and authoritative.
- Typed inter-stage build contract: `AdapterPipelineState` handoffs become typed fields with per-stage postconditions; `extras` demoted to private scratch; one override per stage.
- Migrate all 5 live experiment YAMLs (rf_detr ×2, rtdetrv2 ×2, tutorial_dummy_unet). Delete the dead YOLO-era config surface.

## Capabilities

### New Capabilities
- `adapter-selection`: Resolve the adapter by explicit `adapter.name` with fail-fast on unknown/zero/multiple matches (ADR-0001).
- `engine-adapter-config`: Two-namespace experiment config with single-owner settings, `iou_types` task selector, and the cleaned loss vocabulary (ADR-0003, 0004, 0005, 0006).
- `staged-build-pipeline`: Typed inter-stage build contract with enforced per-stage postconditions (ADR-0011).
- `canonical-prediction`: Single prediction shape `{labels, boxes, scores, masks?}` emitted by every model (ADR-0007).
- `checkpoint-loading`: Flow-gated load strictness with honest loaded/skipped/missing reporting (ADR-0008).
- `run-layout-tracking`: Per-run isolated artifact dirs with shared MLflow/TensorBoard and the `ExperimentTracker` contract (ADR-0009, 0010).

### Modified Capabilities
<!-- None: no existing specs in openspec/specs/. -->

## Impact

- **Config (epicenter, ~140 access sites):** `infra/config/schema_*.py`, `infra/config/context.py`.
- **Selection:** `infra/adapter/core/registry.py`, `spec.py`.
- **Build:** `infra/engine/model/wrappers/common/staged_adapter_builder.py`, `model_builder_base.py`.
- **Loss:** `infra/engine/model/losses/criterion_spec_resolver.py`, `adapters/{common,concrete}_adapter.py`.
- **Prediction:** `infra/core/prediction.py`.
- **Checkpoint:** `infra/engine/model/wrappers/common/checkpoint_adapter.py` + flow callers.
- **Tracking/layout:** `infra/cli/run_layout.py`, `infra/tracking/api/factory.py`, `tb_backend.py`, `mlflow_backend.py`, `engine/callbacks_stack/runtime/tracking.py`.
- **Configs:** 5 live experiment YAMLs migrated; dead YOLO-era fields deleted.
- **No-Goals:** no new models/losses/optimizers, no back-compat shims, no MLflow/TB semantic changes, no new test framework (assert-based self-checks / single `test_*.py` only).
