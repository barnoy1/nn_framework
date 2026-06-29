## 1. Phase 1 — Config schema + namespace cutover (High risk; ADR-0001-cfg, 0003, 0004, 0005, 0006)

- [x] 1.1 Define new pydantic `AppConfig` with top-level `adapter` and `engine` only; `engine.{train,data,execution}`. Split into sub-packages (`infra/config/engine/`, `infra/config/adapter/`) if any schema file exceeds the 200-line cap.
- [x] 1.2 Move settings to single owners: `engine.data.num_classes`, `sync_bn` under `engine.train`, `dn_num_group` under `adapter.model`, `output_dir` owned by CLI→`engine.execution`. Remove duplicated `epochs`/`batch_size`/`num_workers`/`seed` from the old `runtime`.
- [x] 1.3 Make scheduler `T_max` and the training loop both read `engine.train.epochs` (single source).
- [x] 1.4 Add explicit `optimizer{type,...}` and `scheduler{type,...}` blocks under `engine.train`.
- [x] 1.5 Make `data.iou_types ⊆ {bbox, segm}` the sole task selector; delete `data.task`; derive `evaluator.iou_types`; gate mask load/retain/predict/metrics on `"segm" in iou_types`.
- [x] 1.6 Rename loss groups `adapter_common→model_agnostic`, `concrete_model→model_specific`; delete `box/cls/dfl/custom`, `default_yolov11_criterion_pairs`, `normalize_dual_groups`.
- [x] 1.7 Mechanical rename of the ~140 access sites from `app_config.{train,data,runtime,model}` to the new paths.
- [x] 1.8 Make loading the old config shape raise (no deprecation aliases / silent fallback).
- [x] 1.9 Migrate the 5 live experiment YAMLs (rf_detr ×2, rtdetrv2 ×2, tutorial_dummy_unet) to the new shape.
- [x] 1.10 **Check:** config-parse self-check loads all 5 YAMLs and resolves `engine.train`/`adapter.model`; grep gate asserts zero stale `app_config.{train,data,runtime,model}` / `runtime.common.epochs` / `train.epochs` matches.

## 2. Phase 2 — Adapter selection by name (Low risk; ADR-0001-code)

- [x] 2.1 Key the registry on `adapter.name`; remove `matches_source_root` substring logic from `registry.py` / `spec.py`.
- [x] 2.2 Raise a hard error on unknown / zero / multiple adapter matches.
- [x] 2.3 **Check:** `test_registry` — known name resolves; unknown raises; ambiguous raises.

## 3. Phase 3 — Typed inter-stage build contract (Medium risk; ADR-0011)

- [x] 3.1 Replace `AdapterPipelineState` handoffs with typed fields: `model_config`, `config_payload`, `model_api`, `runtime_args`, `model_factory`, `criterion_factory`, `model`, `criterion`, `postprocessor`; demote `extras` to private scratch.
- [x] 3.2 Enforce per-stage postconditions; head stage must end with `model`/`criterion`/`postprocessor` set; collapse `overrides_by_stage` to one override per stage.
- [x] 3.3 **Check:** build smoke — a stage that omits a required typed field raises at the stage boundary.

## 4. Phase 4 — Loss resolver rename + capability gating (Low risk; ADR-0002, 0004)

- [x] 4.1 Update `criterion_spec_resolver.py` and `adapters/{common,concrete}_adapter.py` to the `model_agnostic`/`model_specific` vocabulary; allow empty/partial `model_agnostic`; let unsupported agnostic losses be silently inactive.
- [x] 4.2 **Check:** `test_loss_split` — agnostic/specific routing documented and asserted; routing regression fails.

## 5. Phase 5 — Canonical prediction shape (Low risk; ADR-0007)

- [x] 5.1 Make the postprocessor return `{labels, boxes, scores, masks?}` (`masks` iff `segm ∈ iou_types`); remove the 4-branch `to_result_list` shim in `infra/core/prediction.py`.
- [x] 5.2 **Check:** `test_prediction_shape` — detection omits `masks`, segmentation includes it.

## 6. Phase 6 — Flow-gated checkpoint strictness (Medium risk; ADR-0008)

- [x] 6.1 Train flow: permissive load + loud loaded/skipped/missing summary.
- [x] 6.2 Eval/inference flows: strict fail-fast on missing/shape-mismatched core weights; add `--allow-partial` opt-out; stop `safe_load_state_dict` from silently skipping.
- [x] 6.3 Implement `validate_checkpoint_class_compatibility`; head class-count mismatch in eval is always a hard error.
- [x] 6.4 **Check:** `test_checkpoint_strict` — eval mismatch raises (even with `--allow-partial` for class count); train mismatch warns and proceeds.

## 7. Phase 7 — Per-run dirs + shared tracking (Medium risk; ADR-0009, 0010)

- [x] 7.1 Write heavy artifacts to `<output_base>/runs/<run_id>/{checkpoint,best,configs,logs,inference}`; create the whole tree once in `prepare_run_layout`.
- [x] 7.2 Share `<output_base>/mlflow` (one store, keyed by MLflow run_id) and per-run `<output_base>/tensorboard/<run_id>`.
- [x] 7.3 Rename the visualization logger to `ExperimentTracker`; config under `engine.execution.tracking`; `log_artifact` persists on MLflow / best-effort preview on TB; add `log_execution_config` to the protocol. Keep MLflow default-on and authoritative.
- [x] 7.4 **Check:** 2-run smoke — two MLflow runs in one store, independent `runs/<run_id>/checkpoint`, per-run TB subdir, artifact retrievable, metric-key set matches `main`.

## 8. Cleanup

- [x] 8.1 Delete the dead YOLO-era config/loss surface left after phases 1–7.
- [x] 8.2 Update `docs/adapter_tutorial/` and adapter READMEs to reference the new `adapter`/`engine` config and typed build contract.

## 9. Acceptance gate (end-to-end)

- [ ] 9.1 **Must pass after all phases:** the rf_detr instance-seg train flow runs clean on the migrated config:
  `python cli.py train --config infra/adapter/rf_detr/experiments/rfdetr_small_coco_instance_seg_rle_1ch.yaml --output-dir /home/ronbar/repo/nn_framework/out`
  — config parses under the new `adapter`/`engine` shape, adapter resolves by `adapter.name`, the typed build pipeline completes, `segm`-gated masks flow, and per-run artifacts land under `out/runs/<run_id>/...` with MLflow/TB written. Run ≥1 epoch (or a short smoke override) to exit 0.
