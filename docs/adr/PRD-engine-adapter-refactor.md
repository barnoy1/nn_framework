# PRD — Engine/Adapter Config & Architecture Refactor (11 ADRs)

**Audience:** engineering team executing the refactor.
**Status:** Draft for execution. **Source of truth:** `docs/adr/0001..0011`, `CONTEXT.md`.
**Primary constraint:** **Safe AND complete** — structure-only refactor; the *only* intended
behavior changes are three documented bug fixes (below). MLflow/TensorBoard output stays
functionally identical.

---

## 1. Executive Summary

**Problem.** `nn_framework` was reverse-engineered from YOLO, then grew an adapter layer.
The result is a config and build surface with overlapping ownership: the word "adapter" is
overloaded, the framework/adapter boundary is invisible in config, task selection and loss
groups have dead/duplicated fields, and several values have two sources of truth — one of
which silently corrupts training (LR anneals over 120 epochs while the loop runs 250).

**Solution.** Reorganize the experiment config into two explicit namespaces — `adapter`
(model-owned) and `engine` (framework-owned) — with exactly one owner per setting, and make
four implicit contracts explicit and fail-fast: adapter selection, the staged build pipeline,
the prediction shape, and checkpoint-load strictness. Delete the dead YOLO-era surface.

**Success criteria (measurable):**
1. **One source of truth:** scheduler `T_max` and the training loop read the same
   `engine.train.epochs`; a grep finds zero remaining `runtime.common.epochs` / `train.epochs`
   split reads.
2. **Parse-clean cutover:** all 5 live experiment YAMLs load into the new `AppConfig` with
   zero deprecation aliases; loading the *old* shape raises (no silent fallback).
3. **Tracking parity:** a 2-epoch smoke run produces MLflow runs + TensorBoard scalars
   byte-comparable in structure to `main` (same metric keys, artifacts retrievable).
4. **No-clobber runs:** two consecutive runs keep independent `runs/<run_id>/checkpoint`
   while sharing one MLflow store and one TB logdir.
5. **Every non-trivial change ships one runnable check** (assert-based `__main__` or one
   `test_*.py`); all checks green.

---

## 2. Functional Scope

### Personas
- **Framework engineer** — owns `engine.*` (training loop, data, execution, tracking).
- **Adapter author** — owns `adapter.*` (model definition, losses, postprocessor); enables a
  new model by config + an adapter package, never by touching framework internals.

### User stories & acceptance criteria

**S1 — Unambiguous adapter selection (ADR-0001)**
*As an adapter author, I select my adapter by an explicit name so renaming a vendored
`raw_models/` dir never silently mis-routes.*
- AC: `adapter.name` resolves the adapter; unknown name → hard error; config matching zero or
  multiple adapters → hard error. `matches_source_root` substring logic removed.

**S2 — Visible framework/adapter boundary (ADR-0005, 0006)**
*As any engineer, I see ownership at a glance.*
- AC: top-level keys are exactly `adapter` and `engine`. `engine.train` (math + explicit
  `optimizer{type,...}` / `scheduler{type,...}` blocks), `engine.data`, `engine.execution`
  (renamed from `runtime`). Duplicated `epochs`/`batch_size`/`num_workers`/`seed` removed from
  the old `runtime`. Each setting has one owner: `engine.data.num_classes`, `sync_bn` under
  `engine.train`, `dn_num_group` under `adapter.model`, `output_dir` owned by CLI→`engine.execution`.

**S3 — Single task selector (ADR-0003)**
*As an engineer, one switch decides detection vs instance-seg.*
- AC: `data.iou_types ⊆ {bbox, segm}` is the sole selector; `data.task` deleted;
  `evaluator.iou_types` derived from `iou_types`. `"segm" in iou_types` gates mask
  load/retain/predict/metrics.

**S4 — Clean loss vocabulary (ADR-0002, 0004)**
*As an adapter author, loss groups match the glossary and tolerate model capability.*
- AC: legacy `box/cls/dfl/custom`, `default_yolov11_criterion_pairs`, and `normalize_dual_groups`
  cross-population deleted. Groups renamed `adapter_common→model_agnostic`,
  `concrete_model→model_specific`. Empty/partial `model_agnostic` is valid; an unsupported
  agnostic loss is silently inactive (documented contract, not validated).

**S5 — Canonical prediction (ADR-0007)**
*As a consumer, every model emits the same result shape.*
- AC: postprocessor (head stage) returns `{labels, boxes, scores, masks?}`, `masks` present
  iff `segm ∈ iou_types`. The 4-branch `to_result_list` shim removed.

**S6 — Honest checkpoint loading (ADR-0008)**
*As an engineer, a half-loaded model never silently produces wrong metrics.*
- AC: train flow → partial load allowed + loud loaded/skipped/missing summary. Eval/inference
  → strict, fail-fast on missing/shape-mismatched core weights; `--allow-partial` opt-out for
  debugging. Head class-count mismatch in eval → always hard error. No-op
  `validate_checkpoint_class_compatibility` implemented; `safe_load_state_dict` no longer
  silently skips.

**S7 — Isolated runs, shared tracking (ADR-0009, 0010)**
*As an engineer, consecutive runs don't clobber, but one UI compares all runs.*
- AC: heavy artifacts → `<output_base>/runs/<run_id>/{checkpoint,best,configs,logs,inference}`;
  shared `<output_base>/mlflow` (one store, keyed by MLflow run_id) and
  `<output_base>/tensorboard/<run_id>`. Whole tree created once in `prepare_run_layout`.
  "Visualization logger" renamed `ExperimentTracker`; config `engine.execution.tracking`;
  `log_artifact` = persist on MLflow / best-effort non-authoritative preview on TB;
  `log_execution_config` in the protocol. MLflow stays default-on and authoritative.

**S8 — Explicit build pipeline (ADR-0011)**
*As a framework engineer, a missing inter-stage handoff fails at definition, not stages later.*
- AC: `AdapterPipelineState` handoffs become typed fields (`model_config`, `config_payload`,
  `model_api`, `runtime_args`, `model_factory`, `criterion_factory`, `model`, `criterion`,
  `postprocessor`); `extras` demoted to private scratch. Per-stage postconditions enforced;
  head ends with model/criterion/postprocessor set. `overrides_by_stage` collapsed to one
  override per stage.

### Non-Goals
- No new model support, no new loss, no optimizer beyond making the existing one explicit.
- No deprecation/back-compat shims (no external configs exist).
- No change to MLflow/TB *semantics* — wiring only.
- No new test framework; assert-based self-checks / single `test_*.py` only.
- No abstraction added beyond what an ADR names (ponytail).

---

## 3. Technical Specification

### Architecture (target)
```
experiment.yaml
├── adapter:   name, model{source_root, config_path, num_classes, num_queries,
│              hidden_dim, dn_num_group, losses{model_agnostic, model_specific}}
└── engine:
    ├── train:      epochs, batch_size, lr, optimizer{type,...}, scheduler{type,...},
    │               ema, precision, sync_bn
    ├── data:       num_classes, iou_types, loaders, augments
    └── execution:  output_dir, flows, device, tracking{backends, ...}

build:  config → runtime → weights(construct model) → head(postprocessor)   [typed state]
flow:   mangr_{train,eval,inference} → build_flow_runtime → registry(adapter.name)
                                     → BuiltComponents → loop reads engine.train
output: <output_base>/{runs/<run_id>/..., mlflow, tensorboard/<run_id>}
```

### Integration points
- **Config:** `infra/config/schema_*.py` (pydantic) — the cutover epicenter (~140 access sites).
- **Selection:** `infra/adapter/core/registry.py`, `spec.py`.
- **Build:** `infra/engine/model/wrappers/common/staged_adapter_builder.py`, `model_builder_base.py`.
- **Loss:** `infra/engine/model/losses/criterion_spec_resolver.py`, `adapters/{common,concrete}_adapter.py`.
- **Prediction:** `infra/core/prediction.py`.
- **Checkpoint:** `infra/engine/model/wrappers/common/checkpoint_adapter.py` + flow callers.
- **Tracking/layout:** `infra/cli/run_layout.py`, `infra/tracking/api/factory.py`, `tb_backend.py`,
  `mlflow_backend.py`, `engine/callbacks_stack/runtime/tracking.py`.
- **Configs to migrate (5):** rf_detr ×2, rtdetrv2 ×2, tutorial_dummy_unet.

### Constraints
- File cap 200 lines, ≤5 files/dir, SOLID — if a schema split exceeds the cap, factor into a
  sub-package (e.g. `infra/config/engine/`, `infra/config/adapter/`).
- Trust-boundary validation (config parse, checkpoint compatibility) is **not** simplified away.

---

## 4. Risks & Rollout

### Phased rollout (dependency-ordered — see `plan.md` for task IDs)
| Phase | Scope | ADRs | Risk |
|-------|-------|------|------|
| 1 | Config schema + ~140-site mechanical rename + migrate 5 YAMLs | 0001(cfg),0003,0004,0005,0006 | **High** (wide) |
| 2 | Registry selection by name | 0001(code) | Low |
| 3 | Typed inter-stage build contract | 0011 | Medium |
| 4 | Loss resolver rename + capability gating | 0002,0004 | Low |
| 5 | Canonical Prediction; drop shim | 0007 | Low |
| 6 | Flow-gated checkpoint strictness | 0008 | Medium |
| 7 | Per-run dirs + shared tracking + ExperimentTracker | 0009,0010 | Medium (don't break MLflow) |

Order: 1→2→3→4, then 5/6/7 independent.

### Technical risks & mitigations
- **R1 — Phase-1 mechanical rename misses a site** → runtime `AttributeError`.
  *Mitigation:* the parse-clean check (load all 5 YAMLs + resolve `engine.train`/`adapter.model`)
  plus a grep gate for stale `app_config.{train,data,runtime,model}` paths.
- **R2 — Silent MLflow/TB regression** (ADR-0009/0010 are the only "live IO" changes).
  *Mitigation:* 2-run smoke test asserting two MLflow runs in one store + per-run TB subdir +
  artifact retrievable; compare metric-key set against `main`.
- **R3 — Checkpoint strictness blocks a legitimate warm-start.**
  *Mitigation:* train flow stays permissive by design; `--allow-partial` escape hatch; the
  strict path is eval/inference only.
- **R4 — Capability-gated silent loss inactivity hides a real typo (ADR-0002).**
  *Accepted* per ADR; the loss-split check documents expected agnostic/specific routing so a
  routing regression still fails.

### Per-phase verification (ponytail one-check rule)
1 config-parse · 2 `test_registry` · 3 build smoke (typed-field error surfaces early) ·
4 `test_loss_split` · 5 `test_prediction_shape` · 6 `test_checkpoint_strict` · 7 2-run smoke.

---

## Open questions
- **Q1:** Confirm `engine.execution` as the final name for the renamed `runtime` section
  (defaulted in ADR-0006; freeing "runtime" for the live model-side execution graph).
- **Q2:** Should this PRD live in `docs/` (version-controlled) or stay a session artifact?
