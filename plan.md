# Wrapper Abstraction & Config Propagation Plan

## Goal

Implement a strict abstraction layer between `nn_framework` and concrete model code so the framework interacts only through a stable interface, while keeping the concrete wrapper lean.

Additional mandatory goal:
- every user tweak/adjustment (losses, coefficients, thresholds, training knobs) must be controlled from `experiment.yaml` and propagated reliably through the framework wrapper into the concrete model wrapper.

---

## SOLID Design Policy (governing this plan)

### S — Single Responsibility Principle
- Framework adapter orchestrates only framework-level wrapper lifecycle.
- Concrete wrapper implements only model-specific behavior.
- Config loading/validation, runtime orchestration, and concrete model mechanics remain separated.

### O — Open/Closed Principle
- Add new model backends by adding a new concrete `nn_wrapper` implementation, without modifying framework flow logic.
- Interface extensions are additive and versioned; existing integrations remain stable.

### L — Liskov Substitution Principle
- Any concrete wrapper implementation can replace another if it satisfies the same adapter contract.
- Flows/trainers must not rely on backend-specific side behavior outside the contract.

### I — Interface Segregation Principle
- Keep wrapper contract focused on required capabilities only:
  - component build,
  - checkpoint operations,
  - dn-group configuration.
- Avoid forcing concrete wrappers to implement unrelated optional behavior.

### D — Dependency Inversion Principle
- Framework depends on abstractions (`ModelWrapperAdapter`, `ModelBuilder`, `CheckpointAdapter`, `DnGroupConfigurer`), not concrete RT-DETR classes.
- Concrete wrappers are loaded as plugins through factory/loader abstractions.

---

## Requirements Coverage

1. Thorough analysis of current implementation.
2. Keep concrete wrapper minimal; move model-agnostic behavior to framework-side base wrapper.
3. Use adapter as the single entrypoint between framework and concrete model wrapper.
4. Framework abstract adapter exposes interface populated by concrete wrapper implementation.
5. Framework/model interaction occurs only via this interface.
6. `experiment.yaml` is the single user-facing source for model tweaks and coefficients; no hidden hardcoded override paths.

---

## Current-State Analysis (as implemented now)

### Framework side
- `infra/engine/model/base.py`
  - defines `BuiltComponents`, `ModelBuilder`, `CheckpointAdapter`, `DnGroupConfigurer`, `ModelWrapperAdapter` (currently as `Protocol`).
- `infra/engine/model/wrapper_creators.py`
  - loads concrete modules dynamically and instantiates wrapper.
  - still has RT-DETR-specific fallback class naming.
- `infra/engine/model/adapter.py`
  - currently empty; should become framework-side adapter entrypoint.
- `infra/engine/flows/common/runtime.py`, `trainer.py`, `mangr_train.py`, `mangr_eval.py`
  - already consume wrapper methods, but contract hardening is still needed.

### Concrete RT-DETR side
- `raw_models/RT-DETR/rtdetrv2_pytorch/nn_wrapper/adapter.py`
  - composes builder + checkpoint and delegates methods.
  - includes path bootstrapping and direct module composition.
- `nn_wrapper/builder.py`
  - builds model components and applies RT-DETR-specific config/loss/optimizer behavior.
- `nn_wrapper/checkpoint.py`
  - checkpoint resolution + compatibility validation + safe load.

### Gap summary
- Abstraction exists conceptually but is not fully centralized in framework adapter layer.
- Some factory/loader behavior is still model-specific.
- `experiment.yaml` propagation exists partially, but there is no explicit end-to-end contract that guarantees every user tweak reaches concrete model build/checkpoint/runtime behavior.

---

## Target Architecture

### 1) Single framework adapter entrypoint
- Implement `infra/engine/model/adapter.py` as the canonical framework adapter orchestrator.
- It owns agnostic delegation and lifecycle orchestration.
- All framework code imports/uses this adapter path only.
- SOLID alignment:
  - S: one orchestration responsibility.
  - D: framework layer consumes abstractions only.

### 2) Strict interface contract
- Keep interface in `infra/engine/model/base.py` as the only contract surface.
- Prefer explicit abstract class semantics (or strict runtime validation if `Protocol` is retained).
- Required methods:
  - `build_components()`
  - `configure_fixed_dn_num_group(...)`
  - `load_checkpoint_state(...)`
  - `validate_checkpoint_class_compatibility(...)`
  - `safe_load_state_dict(...)`
- SOLID alignment:
  - L: each concrete wrapper must be substitutable through this contract.
  - I: only essential methods are required.

### 3) Lean concrete wrappers
- Concrete `nn_wrapper/adapter.py` remains minimal:
  - binds concrete builder/checkpoint/configurer.
  - contains no framework-agnostic orchestration logic.
  - contains no duplicated generic flow behavior.
- SOLID alignment:
  - S: concrete adapter handles only model-specific composition.
  - O: new model wrapper implementations extend behavior without framework edits.

### 4) YAML-first propagation model
- User changes are authored only in `experiment.yaml`.
- Propagation chain is explicit and testable:
  - `experiment.yaml`
  - framework config loader (`AppConfig`/runtime config)
  - framework adapter contract payload
  - concrete wrapper builder/checkpoint/runtime knobs
  - model/criterion/postprocessor/optimizer behavior

### 6) Production configuration governance
- Treat `experiment.yaml` as a versioned contract, not just input data.
- Add config contract version field (example: `meta.config_contract_version`) and validate it at load time.
- Enforce strict schema mode for model-facing sections (`model`, `model.losses`, `train`, `data`, `runtime.common`, `runtime.actions`).
- Any unknown key under model-facing sections must fail fast with a path-aware error.

---

## Production experiment.yaml Propagation Matrix (reference-aligned)

This matrix is based on the production reference config `experiment/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yaml` and is the source of truth for wrapper propagation.

### A) `model.*` (concrete-model build contract)
- `model.source_root`
  - Owner: framework runtime model root resolver.
  - Consumer: wrapper module loader.
  - Rule: path must exist and contain `nn_wrapper` package.
- `model.model_config_path`, `variant`, `num_classes`, `num_queries`, `hidden_dim`, `sync_bn`, `dn_num_group`
  - Owner: AppConfig model schema.
  - Consumer: concrete model builder + dn-group configurer.
  - Rule: all model-shape-impacting fields validated before checkpoint load.

### B) `model.losses.criterion_pairs.*` (critical tuning path)
- Groups: `box`, `cls`, `dfl`, `custom`.
- Entry shape: `{loss: <name_or_prefix>, coef: <float>}`.
- Owner: AppConfig model loss schema.
- Consumer: concrete loss controller in wrapper builder.
- Rules:
  - `coef` must be finite numeric and non-negative.
  - Prefix losses (for example `loss_bbox_aux_`) must expand deterministically across indexed heads.
  - Expansion must be logged (resolved key list + effective coefficient).
  - Unresolved patterns must fail (or warn+fail in strict mode).

### C) `train.*` (agnostic optimizer/runtime knobs)
- Fields: `epochs`, `batch_size`, `val_batch_size`, `num_workers`, `lr`, `backbone_lr_multiplier`, `weight_decay`, `grad_clip_norm`, `mixed_precision`, `use_ema`, `ema_decay`, `ema_warmup_updates`, `seed`, `val_every_n_epochs`, `save_every_n_epochs`.
- Owner: framework train schema.
- Consumer: framework trainer + concrete optimizer factory where model-specific grouping is required.
- Rule: no concrete default may override explicit train value from yaml.

### D) `data.*` (dataset/eval semantics)
- Fields: `mapping`, `label2classid`, `class_id_to_name`, `num_classes`, evaluator config.
- Owner: data schema.
- Consumer: loader/runtime + eval diagnostics + postprocessor label rendering.
- Rule: mapped label IDs must be validated against effective model classes and surfaced before training/eval.

### E) `runtime.common.*` and `runtime.visualization.*` (flow behavior)
- Fields include `score_threshold`, `batch_size`, `num_workers`, visualization sample count.
- Owner: runtime schema.
- Consumer: eval/inference/export flows.
- Rule: standalone flow defaults must come from yaml unless explicitly overridden by CLI.

### F) `runtime.actions.*` (action entrypoint defaults)
- Fields: train/eval/inference checkpoint defaults, export paths, inference input dirs, export-coco-rle parameters.
- Owner: runtime action schema + CLI defaults resolver.
- Consumer: action parsers and command handlers.
- Rule: action defaults must be resolved from yaml once, then passed in-process to invoke handlers without mutation side paths.

### 5) No direct framework dependency on concrete symbols
- Remove concrete-name fallbacks like `RTDETRv2WrapperAdapter` from generic loader paths.
- Require canonical factory exports in concrete modules.
- SOLID alignment:
  - D: no direct dependency on concrete symbol names.
  - O: plugin model for new wrappers.

---

## Detailed Migration Plan

### Phase 0 — Contract freeze and inventory
1. Freeze and document interface in `infra/engine/model/base.py`.
2. Create a config propagation matrix in this file (or sibling doc section):
   - each tweak key from `experiment.yaml`,
   - owning `AppConfig` field,
   - consuming concrete component (`builder`, `loss controller`, `optimizer`, `postprocessor`, etc.),
   - validation rule/default.
3. Mark unsupported keys explicitly (fail-fast with actionable error).
4. Add strict handling mode for production runs:
  - `strict_config=true` (default for CI/prod),
  - unknown model-facing keys become hard errors.

Deliverable:
- stable contract + propagation matrix.
- SOLID gate:
  - verify interface is minimal (I) and substitutable (L).

### Phase 1 — Implement framework-side agnostic adapter
1. Implement `FrameworkModelAdapter` in `infra/engine/model/adapter.py`.
2. Constructor accepts pluggable concrete collaborators:
   - `ModelBuilder`
   - `CheckpointAdapter`
   - `DnGroupConfigurer`
3. Adapter implements full wrapper interface and delegates behavior.
4. Add strict runtime conformance checks for concrete collaborators.

Deliverable:
- framework owns generic orchestration.
- SOLID gate:
  - verify framework adapter has one reason to change (S).

### Phase 2 — Slim concrete RT-DETR adapter
1. Refactor `nn_wrapper/adapter.py` to only wire concrete collaborators.
2. Move any agnostic logic currently in concrete adapter into framework adapter.
3. Keep RT-DETR-specific implementation in:
   - `nn_wrapper/builder.py`
   - `nn_wrapper/checkpoint.py`

Deliverable:
- concrete wrapper reduced to minimal implementation glue.
- SOLID gate:
  - verify no agnostic logic remains in concrete wrapper (S, D).

### Phase 3 — Unify loader/factory entrypoints
1. In `infra/engine/model/wrapper_creators.py`, require one canonical factory API from concrete wrapper modules.
2. Remove concrete class-name fallback assumptions.
3. Use `infra/engine/model/adapter.py` as the sole framework assembly entrypoint.

Deliverable:
- model-agnostic dynamic loading.
- SOLID gate:
  - verify adding a new backend requires no changes to framework flows (O, D).

### Phase 4 — Enforce experiment.yaml propagation end-to-end
1. Define explicit mapping for tweak categories:
   - loss enables/weights/coefs,
   - optimizer/scheduler knobs,
   - postprocessor thresholds,
   - model-specific toggles.
2. Ensure mapping path is deterministic:
   - `experiment.yaml -> AppConfig -> adapter -> concrete builder/controller`.
3. Add fail-fast checks:
   - unknown tweak key,
   - invalid type/range,
   - non-propagated key detected.
4. Remove hidden defaults that bypass `experiment.yaml` unless documented as immutable framework defaults.
5. Add propagation observability:
  - on startup emit a compact “effective model tweak table” including resolved `criterion_pairs`.
  - store this table under run artifacts (`configs/effective_model_tweaks.yaml`).
6. Add precedence policy (must be deterministic and documented):
  - CLI override > experiment.yaml > schema default.
  - no additional precedence layers.

Deliverable:
- all user-facing tweaking exposed and propagated through architecture.

### Phase 5 — Framework call-site hardening
1. Verify all interactions use wrapper interface only:
   - runtime builder,
   - trainer epoch DN config,
   - checkpoint loading in train/eval/inference flows.
2. Eliminate direct concrete module imports from framework runtime paths.

Deliverable:
- strict interface-only framework/model interaction.
- SOLID gate:
  - verify runtime and trainer modules compile/run with abstraction-only dependencies (D, L).

### Phase 6 — Deprecation & compatibility window
1. Keep temporary compatibility exports where needed.
2. Log deprecation warnings for old entrypoints.
3. Remove deprecated paths after validation window.

Deliverable:
- smooth migration without breaking active workflows.

### Phase 7 — Validation & acceptance
1. Contract tests:
   - concrete wrapper satisfies interface.
2. Propagation tests:
   - tweak values from `experiment.yaml` are reflected in concrete runtime objects.
3. Flow smoke tests:
   - train/eval/inference/export paths remain functional.
4. Artifact sanity:
   - outputs reflect configured tweak effects (e.g., loss coefficient changes visible in logged metrics).

Deliverable:
- production-ready abstraction with verified config propagation.

---

## experiment.yaml Propagation Checklist (must pass)

For each user-facing tweak key:
1. Defined in schema/config model.
2. Loaded into `AppConfig` with type validation.
3. Present in adapter-consumed config payload.
4. Consumed by concrete component intentionally.
5. Observable in runtime behavior/logging.
6. Covered by test/smoke assertion.

If any step fails, startup should fail with explicit error indicating missing propagation stage.

Additional production checks:
7. Every loss key/prefix in `criterion_pairs` resolves to at least one criterion term.
8. Effective resolved coefficients are persisted to run artifacts.
9. Checkpoint compatibility checks use effective `num_classes` after mapping/overrides are finalized.

---

## Implementation Notes from Reference YAML

- Keep `model.losses.criterion_pairs.custom` fully model-specific; framework only validates structure/types and propagation integrity.
- Keep class mapping behavior explicit:
  - dataset contiguous labels -> mapped model class IDs (`data.mapping`) -> reporting labels (`class_id_to_name`).
- Runtime actions in yaml should remain the only default source for checkpoints/paths used by CLI actions.
- Preserve in-process invocation path (no subprocess boundary) so config object continuity is guaranteed from parser -> runtime -> wrapper.

---

## Risks & Mitigations

- Risk: silent config drops (key exists but not consumed).
  - Mitigation: propagation matrix + fail-fast unknown/non-routed key checks.

- Risk: abstraction leakage through concrete class references.
  - Mitigation: canonical factory contract + no concrete-name fallbacks.

- Risk: migration regressions in training/eval flows.
  - Mitigation: phased rollout + smoke tests after each phase.

---

## Definition of Done

- Framework uses a single agnostic adapter entrypoint.
- Concrete wrapper is lean and model-specific only.
- Interface contract is the only interaction surface.
- All user tweaks are controlled from `experiment.yaml` and fully propagated to concrete model behavior through wrapper architecture.
- Train/eval/inference/export flows pass smoke validation with no contract breakage.

Additionally, SOLID acceptance must hold:
- S: each adapter layer has one bounded responsibility.
- O: new model backend added via plugin wrapper only.
- L: concrete wrappers are interchangeable under the contract.
- I: interface remains capability-focused and minimal.
- D: framework code depends only on abstractions, never concrete backend classes.

---

# Dual-Criterion Loss Plan (Agnostic + Concrete)

## Goal

Introduce two coordinated criteria in the training/eval loss path:
1. an agnostic adapter criterion for YOLO-like/common losses,
2. a concrete adapter criterion for model-specific losses,

with deterministic fallback behavior:
- if a loss is not found in experiment config, use model default coefficient.

## Current behavior snapshot

- Framework currently executes a single criterion in trainer paths (`trainer.criterion(outputs, targets)`).
- Criterion preparation/coefficient enablement is framework-owned via `prepare_base_criterion_for_agnostic_flow(...)` in `infra/engine/model/losses/composite_criterion.py`.
- Loss splitting/reporting (`box/cls/dfl/custom`) is already centralized in framework (`infra/engine/training/loss_components.py`).

## Target behavior

- Keep trainer call shape unchanged (single callable criterion from framework POV).
- Internally compose two criterion handlers:
  - **AgnosticYoloCriterionAdapter**: owns common losses (`loss_bbox`, `loss_giou`, `loss_vfl`/`loss_focal`, optional `loss_dfl`).
  - **ConcreteCriterionAdapter**: owns model-specific variants (`*_aux_*`, `*_dn_*`, `*_enc_*`, and other backend-specific keys).
- Merge outputs into one loss dict for backward-compatible logging/metrics/callbacks.

## File structure plan (SOLID-aligned)

Create explicit criterion modules split by responsibility:

- Framework-common (agnostic, reusable):
  - `infra/engine/training/criteria/common/yolo_common_criterion.py`
    - implements YOLO-like/common loss handling only.
  - `infra/engine/training/criteria/common/spec_resolver.py`
    - resolves configured coefficients + fallback-to-default policy.
  - `infra/engine/training/criteria/common/composite_criterion.py`
    - orchestrates multiple criterion adapters and merges results.

- Adapter/concrete (model-specific extension point):
  - `raw_models/RT-DETR/rtdetrv2_pytorch/nn_wrapper/criteria/concrete_criterion_adapter.py`
    - applies RT-DETR-specific loss logic and variant keys.

- Optional shared interface contract:
  - `infra/engine/training/criteria/base.py`
    - defines criterion adapter protocol/ABC used by common + concrete adapters.

SOLID ownership rules:
- **S (Single Responsibility):** each criterion file handles one concern (common, concrete, resolver, composite).
- **O (Open/Closed):** new models add only new concrete adapter files under their wrapper path.
- **L (Liskov):** all adapters satisfy one criterion adapter contract and are interchangeable in composite.
- **I (Interface Segregation):** adapter interface exposes only `compute(outputs, targets)` + metadata needed for merge.
- **D (Dependency Inversion):** composite depends on adapter abstraction, not RT-DETR concrete classes.

## Configuration plan

Extend `model.losses` schema to support explicit dual groups:

- `criterion_pairs.adapter_common`
  - common/agnostic loss specs
- `criterion_pairs.concrete_model`
  - backend-specific loss specs
- `fallback_to_model_default: true`
  - when loss key not explicitly configured, resolve to concrete model criterion default (`weight_dict`).

Backward compatibility:
- keep reading existing `criterion_pairs.{box,cls,dfl,custom}` for one migration window;
- map legacy groups to the new dual groups deterministically.

## Resolution & precedence policy

For each effective loss key (including indexed suffixes):

1. `concrete_model` explicit match (exact/prefix) wins.
2. else `adapter_common` explicit match.
3. else if `fallback_to_model_default=true`, use concrete criterion `weight_dict[key]` or base-key fallback.
4. else coefficient = `0.0`.

Matching semantics:
- exact name matches exact key,
- names ending with `_` are treated as prefix patterns.

## Implementation phases

### Phase 1 — Schema
- Update `infra/config/schema_model.py` with dual criterion structures and fallback flag.
- Add validators for finite, non-negative coefficients.

### Phase 2 — Resolver
- Add a shared loss-spec resolver in framework training/model layer to:
  - normalize aliases,
  - resolve coefficient by precedence,
  - expose debug map: `loss_key -> {source, coef}`.

### Phase 3 — Dual adapters
- Add `AgnosticYoloCriterionAdapter` in `infra/engine/training/criteria/common/yolo_common_criterion.py`.
- Add `ConcreteCriterionAdapter` in `raw_models/RT-DETR/rtdetrv2_pytorch/nn_wrapper/criteria/concrete_criterion_adapter.py`.
- Both return loss dicts with stable key naming.

### Phase 4 — Composite criterion
- Add `CompositeCriterion` callable in `infra/engine/training/criteria/common/composite_criterion.py` that executes both adapters and returns merged dict.
- Conflict policy: deterministic (prefer concrete on exact duplicate key).

### Phase 5 — Builder integration
- In RT-DETR `nn_wrapper/builder.py`, continue building native criterion.
- Wrap/augment with composite criterion and return through `BuiltComponents.criterion`.
- Keep trainer/eval code unchanged.

### Phase 6 — Logging/reporting compatibility
- Preserve existing metric keys (`train/criterion/*`, `val/criterion/*`).
- Optionally add diagnostic artifact showing resolved source (`common` vs `concrete` vs `default`).

### Phase 7 — Validation tests
- Add focused tests for:
  - precedence correctness,
  - prefix expansion behavior,
  - fallback-to-default behavior,
  - merge stability with overlapping keys.

### Phase 8 — YAML migration
- Update experiment YAML examples to the new structure.
- Keep legacy format accepted with deprecation warning for migration window.

## Definition of done (for this feature)

- Two-criterion architecture is active behind one trainer criterion call.
- Agnostic/common and concrete/model-specific losses are both executed.
- Missing config losses correctly fallback to model defaults.
- Existing training/eval logging remains backward compatible.
- New schema + resolver + tests pass and are documented.
