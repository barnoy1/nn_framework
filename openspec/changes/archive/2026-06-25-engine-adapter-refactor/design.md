## Context

`nn_framework` is a model-agnostic PyTorch detection/instance-seg training framework reverse-engineered from YOLO, with an adapter layer bolted on later. The config and build surface accumulated overlapping ownership: "adapter" is overloaded, the framework/adapter boundary is invisible in config, and several values have two sources of truth — notably `epochs`, where the scheduler anneals over `runtime.common.epochs` (120) while the loop runs `train.epochs` (250), silently corrupting LR schedules.

The 11 ADRs (`docs/adr/0001..0011`) decide the target. This change executes them as a **structure-only refactor**: the only intended behavior changes are three documented bug fixes (epochs single-source, honest checkpoint loading, no-clobber runs). MLflow/TensorBoard output stays functionally identical. No external configs exist, so there is no back-compat surface to preserve.

Constraints: 200-line file cap, ≤5 files/dir, SOLID. Trust-boundary validation (config parse, checkpoint compatibility) is NOT simplified away. No new test framework — assert-based `__main__` self-checks or a single `test_*.py` per non-trivial change.

## Goals / Non-Goals

**Goals:**
- One owner per setting across an `adapter` / `engine` namespace split; old shape raises on load.
- Four implicit contracts made explicit and fail-fast: adapter selection, staged build, prediction shape, checkpoint strictness.
- Three bug fixes: single-source epochs, flow-gated checkpoint honesty, per-run isolated artifact dirs.
- Delete the dead YOLO-era config/loss surface.
- Every non-trivial change ships one runnable check.

**Non-Goals:**
- No new model support, no new loss, no optimizer beyond making the existing one explicit.
- No deprecation/back-compat shims.
- No change to MLflow/TB semantics — wiring only.
- No abstraction beyond what an ADR names.

## Decisions

**D1 — Two namespaces (`adapter` / `engine`), no aliases (ADR-0005, 0006).**
`engine.{train,data,execution}`. `runtime`→`execution`. Single owner: `engine.data.num_classes`, `sync_bn` under `engine.train`, `dn_num_group` under `adapter.model`, `output_dir` owned by CLI→`engine.execution`. *Alternative:* a compatibility shim mapping old→new keys — rejected; no external configs exist, and a shim would hide the single bug the split exists to kill (the epochs double-source).

**D2 — Selection by `adapter.name` (ADR-0001).** Registry keyed by explicit name; `matches_source_root` substring logic deleted; zero/multiple/unknown → hard error. *Alternative:* keep substring matching with a tiebreaker — rejected; renaming a `raw_models/` dir would still silently mis-route.

**D3 — `iou_types` as the only task selector (ADR-0003).** `data.task` deleted; `evaluator.iou_types` derived; `"segm" in iou_types` gates the whole mask pathway. *Alternative:* keep both `task` and `iou_types` synced — rejected; two selectors is the data-clump that caused the divergence.

**D4 — Loss vocabulary `model_agnostic` / `model_specific` (ADR-0002, 0004).** Legacy `box/cls/dfl/custom`, `default_yolov11_criterion_pairs`, `normalize_dual_groups` deleted. Unsupported agnostic loss is silently inactive — a *documented* contract, not a validated one. *Alternative:* validate agnostic-loss support per model — rejected per ADR-0002; the loss-split self-check still catches routing regressions.

**D5 — Typed `AdapterPipelineState` (ADR-0011).** Inter-stage handoffs become typed fields with per-stage postconditions; `extras` demoted to private scratch; one override per stage. Makes a missing handoff fail at definition, not stages later. *Alternative:* keep the `extras` dict with key asserts — rejected; untyped bag defers the error and hides the contract.

**D6 — Canonical `{labels, boxes, scores, masks?}` (ADR-0007).** `masks` iff `segm ∈ iou_types`; drop the 4-branch `to_result_list` shim.

**D7 — Flow-gated checkpoint strictness (ADR-0008).** Train = permissive + loud summary; eval/inference = strict fail-fast with `--allow-partial`; head class-count mismatch in eval = always hard error. Implement the no-op `validate_checkpoint_class_compatibility`; stop `safe_load_state_dict` silently skipping.

**D8 — Per-run dirs + shared tracking (ADR-0009, 0010).** `<output_base>/runs/<run_id>/...`; shared `mlflow` store + `tensorboard/<run_id>`; created once in `prepare_run_layout`. Rename logger → `ExperimentTracker`; `engine.execution.tracking`; `log_artifact` persists on MLflow / best-effort preview on TB; `log_execution_config` in the protocol. MLflow stays default-on and authoritative.

**D9 — Schema file layout.** If a pydantic schema split exceeds the 200-line cap, factor into sub-packages (`infra/config/engine/`, `infra/config/adapter/`) rather than one fat module.

## Risks / Trade-offs

- **R1 — Phase-1 mechanical rename (~140 sites) misses a site → runtime `AttributeError`.** → Parse-clean check loads all 5 YAMLs and resolves `engine.train`/`adapter.model`; a grep gate fails on any stale `app_config.{train,data,runtime,model}` path.
- **R2 — Silent MLflow/TB regression (D8 is the only live-IO change).** → 2-run smoke test asserts two MLflow runs in one store + per-run TB subdir + retrievable artifact; compare metric-key set against `main`.
- **R3 — Checkpoint strictness blocks a legitimate warm-start.** → Train flow stays permissive by design; `--allow-partial` escape hatch; strict path is eval/inference only.
- **R4 — Capability-gated silent loss inactivity hides a real typo (D4).** → Accepted per ADR-0002; the loss-split self-check documents expected agnostic/specific routing so a routing regression still fails.

## Migration Plan

Dependency-ordered phases (1→2→3→4, then 5/6/7 independent), each with its one check:
1. Config schema + ~140-site rename + migrate 5 YAMLs (0001-cfg,0003,0004,0005,0006) — **High risk**. Check: config-parse.
2. Registry selection by name (0001-code). Check: `test_registry`.
3. Typed inter-stage build contract (0011). Check: build smoke (typed-field error surfaces early).
4. Loss resolver rename + capability gating (0002,0004). Check: `test_loss_split`.
5. Canonical Prediction; drop shim (0007). Check: `test_prediction_shape`.
6. Flow-gated checkpoint strictness (0008). Check: `test_checkpoint_strict`.
7. Per-run dirs + shared tracking + `ExperimentTracker` (0009,0010). Check: 2-run smoke.

Rollback: each phase is an independent commit; no data migration, so revert the commit. Phase 1 is the only wide blast radius — its parse-clean + grep gate must be green before merging downstream phases.

## Open Questions

- **Q1:** Confirm `engine.execution` as the final name for the renamed `runtime` section (defaulted in ADR-0006; frees "runtime" for the live model-side execution graph).
- **Q2:** Should the PRD live in `docs/` (version-controlled) or stay a session artifact? (It currently lives in `docs/adr/`.)
