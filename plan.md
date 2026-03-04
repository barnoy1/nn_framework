# nn_framework SRP Refactor Plan

## Purpose

Refactor the codebase to improve maintainability, scalability, and readability by enforcing SRP and clean architecture boundaries, while preserving existing runtime behavior.

This plan is intentionally **non-breaking first**:
- keep CLI contracts stable,
- keep current flow entrypoints runnable,
- keep exported symbols stable where possible,
- avoid placeholder/ghost interfaces and empty files.

---

## Current Hotspots (from source analysis)

### 1) CLI God Module
- `cli.py` (~512 LOC) currently mixes:
  - argument parsing,
  - config loading/default-resolution,
  - run layout generation,
  - subprocess command orchestration for train/eval/inference/export.

**Impact:** hard to test, hard to evolve commands independently, high coupling between parsing and execution.

### 2) Callback God Module
- `infra/engine/callbacks.py` (~600 LOC) combines unrelated concerns:
  - lifecycle protocol (`Callback`, `CallbackList`),
  - experiment tracking (`MLflowCallback`),
  - checkpointing,
  - augmentation scheduling,
  - visualization generation,
  - YOLO-style metrics/artifact plotting.

**Impact:** violates SRP, large import surface, difficult to reason about changes in one callback without side-effects.

### 3) Runtime Builder Overload
- `infra/engine/flows/common/runtime.py` combines:
  - hydra config loading,
  - data preparation side effects,
  - data loader construction,
  - model wrapper/component assembly.

**Impact:** orchestration and construction logic are intertwined; limited composability for alternate execution modes.

### 4) Eval Shared Module Overload
- `infra/engine/flows/eval/shared.py` (~289 LOC) mixes:
  - dataset sampling,
  - inference loop,
  - visualization writing,
  - metrics and diagnostics assembly.

**Impact:** difficult to unit test individual concerns and extend evaluation pipeline safely.

---

## Refactor Principles

1. **SRP first:** each module should have one reason to change.
2. **Stable external API:** preserve current CLI command names, arguments, and outputs.
3. **No speculative abstractions:** do not add interfaces unless already needed by at least one real consumer.
4. **No ghost files:** every created file must contain real behavior used by runtime code.
5. **Incremental migration:** extract in phases; keep compatibility shims only where necessary.

---

## Target Architecture (incremental)

### A) CLI package decomposition

Create a dedicated package under `infra/cli/`:

- `infra/cli/config_defaults.py`
  - path resolution,
  - yaml payload loading,
  - runtime/action defaults derivation,
  - dataset export defaults helpers.

- `infra/cli/commands.py`
  - command builders/executors:
    - train/eval/inference/export-onnx/export-coco-rle.

- `infra/cli/parser.py`
  - parser registration and argument normalization.

- `infra/cli/run_layout.py`
  - run directory and `execution.yaml` preparation.

- `infra/cli/main.py`
  - top-level orchestration (`main()`).

Then keep root `cli.py` as a thin compatibility entrypoint delegating to `infra.cli.main`.

### B) Callback package decomposition

Split callback concerns into separate modules under a new package:

- `infra/engine/callbacks_base.py`
  - `Callback`, `CallbackList` only.

- `infra/engine/callbacks_tracking.py`
  - `MLflowCallback`.

- `infra/engine/callbacks_checkpoint.py`
  - `CheckpointCallback`.

- `infra/engine/callbacks_training.py`
  - `EMACallback`, `DynamicAugCallback`.

- `infra/engine/callbacks_visualization.py`
  - `ValidationVisualizationCallback`.

- `infra/engine/callbacks_artifacts.py`
  - `YoloStyleArtifactsCallback`.

Keep `infra/engine/callbacks.py` as a temporary re-export module during migration.

### C) Runtime builder separation

Extract from `infra/engine/flows/common/runtime.py`:

- `infra/engine/flows/common/config_loader.py`
  - hydra composition and validation.

- `infra/engine/flows/common/data_runtime.py`
  - `_prepare_data_if_needed`, loader construction.

- `infra/engine/flows/common/model_runtime.py`
  - wrapper creation and component assembly.

`runtime.py` becomes orchestration-only.

### D) Evaluation pipeline separation

Split `infra/engine/flows/eval/shared.py` into:

- `eval_sampling.py` (sample + GT extraction),
- `eval_inference.py` (prediction loop),
- `eval_reporting.py` (json/vis/diagnostics),
- `shared.py` (thin composition facade).

---

## Phase Plan

### Phase 1 (safe, immediate)
1. Decompose `cli.py` into `infra/cli/*` package.
2. Keep `cli.py` as delegating entrypoint only.
3. Verify `python cli.py --help` and command parser behavior.

### Phase 2 (medium risk)
1. Split callback module by concern.
2. Keep old imports stable through `infra/engine/callbacks.py` re-exports.
3. Validate train flow callback execution.

### Phase 3 (medium risk)
1. Split runtime builders into config/data/model construction modules.
2. Keep `build_flow_runtime(...)` signature unchanged.
3. Validate train/eval/inference managers.

### Phase 4 (medium-high risk)
1. Split eval shared pipeline.
2. Preserve `run_eval_artifacts(...)` external contract.
3. Validate metric and artifact parity.

---

## Risk Controls

- Preserve all public function signatures used by entrypoints.
- Add migration via delegation/re-export instead of abrupt renames.
- Avoid changing config schema and hydra paths during structural refactor.
- Execute focused smoke checks after each phase.

---

## Validation Strategy

After each phase:
1. Static check: import key modules and run parser help.
2. Functional smoke:
   - `train` parse path,
   - `eval` parse path,
   - `inference` parse path.
3. Regression check on generated run directory and execution config payload.

---

## Definition of Done

- SRP violations reduced in top hotspots (`cli.py`, callbacks, runtime/eval shared).
- Modules are smaller and concern-focused.
- Existing CLI and flow behavior remains compatible.
- No empty placeholder files introduced.
