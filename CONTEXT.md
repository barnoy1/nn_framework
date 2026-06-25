# nn_framework

A PyTorch object-detection / instance-segmentation training framework. Generic train/eval/inference flows stay model-agnostic; each concrete model repository is integrated behind a single contract.

## Language

**Adapter**:
A package under `infra/adapter/<name>/` that integrates one concrete model repository into the framework, exposing a manifest and a model builder. This is the *only* meaning of "adapter" — the wrapper protocols, loss transformers, and override stages are not "adapters".
_Avoid_: using "adapter" for `ModelWrapperAdapter`/`CheckpointAdapter` (call these wrapper protocols), for loss classes (call these loss transformers / criterion adapters, always qualified), or for override stages.

**Adapter selection**:
Resolving which adapter handles a run. Driven by an explicit `model.adapter` key naming the adapter; zero or multiple matches are hard errors. (See ADR-0001.)
_Avoid_: selecting by `source_root` path substring.

**Model-agnostic loss**:
A base detection loss the framework computes/scales itself — base key in {`loss_bbox`, `loss_giou`, `loss_vfl`, `loss_focal`, `loss_dfl`} with no variant marker. Configured under the `model_agnostic` criterion group.
_Avoid_: "common", "adapter_common", "YOLO loss".

**Model-specific loss**:
A loss owned by the concrete model — any key carrying a variant marker (`_aux_`/`_dn_`/`_enc_`) or explicitly listed in the `model_specific` criterion group. Variant precedence beats the agnostic group, so `loss_bbox_aux_0` is model-specific even though `loss_bbox` is model-agnostic. Configured under the `model_specific` group.
_Avoid_: "concrete", "concrete_model".

**Variant marker**:
A substring (`_aux_`, `_dn_`, `_enc_`) in a loss key marking it as auxiliary / denoising / encoder supervision of a base loss. Presence of any marker forces a loss into the model-specific group.

**Model-agnostic loss support**:
A per-model capability: which of the framework's five model-agnostic losses a given model can actually produce, gated by the model's outputs (e.g. DFL needs distribution outputs). An empty `model_agnostic` group is valid (e.g. a segmentation-only model). An unsupported agnostic loss is silently inactive — never an error. (See ADR-0002.)

**IoU type**:
The canonical task selector, `data.iou_types`, a subset of {`bbox`, `segm`}. The task is *derived* from it: detection = `{bbox}`, instance segmentation = `{bbox, segm}`. Presence of `segm` is the single switch that turns on mask loading, retention, prediction, and metrics. (See ADR-0003.)
_Avoid_: the dead `data.task` field, a separate `evaluator.iou_types` source of truth, or treating "instance segmentation" as a distinct first-class task type.

**Experiment config**:
A single `experiment.yaml` that is *the* contract between framework and adapter (Hydra `@package _global_`, parsed into `AppConfig`), organized into two top-level namespaces that make the boundary visible: `adapter` (everything the adapter/model owns) and `engine` (everything the framework owns). Within `engine`, the `train` section configures the training loop/math (epochs, optimizer, scheduler, EMA, precision) and is the single source of truth for those — never duplicated in `engine.execution`. (See ADR-0005, ADR-0006.)
_Avoid_: putting model-architecture choices under `engine`, training-loop choices under `adapter`, or duplicating a setting across `engine.train` and `engine.execution`.

**Class space (`num_classes`)**:
The number of dataset classes, owned in one place: `engine.data.num_classes`. The adapter reads it to size/realign the model head; pretrained-checkpoint capacity mismatch is handled inside the adapter's head/weights overrides, and the `+1` background slot is a model-internal detail. There is no second authoritative class count in `adapter.model`.
_Avoid_: a separate `model.num_classes` source of truth that can drift from the dataset.

**Config ownership test**:
For any setting, ask "does this describe the model, or how we train/run it?" Model-family architecture knobs (e.g. `dn_num_group`, num_queries, hidden_dim) go under `adapter.model`; training/execution policy (e.g. `sync_bn`, optimizer, scheduler, device) goes under `engine`. A non-DETR adapter simply omits the DETR-only knobs; the engine reads model knobs through the wrapper contract, never reaches into model internals.

**Runtime**:
The live, model-facing execution objects: the adapter's `runtime/` subpackage that binds the concrete model's API, the `runtime` build stage that populates it, `runtime_args`, and the per-run object graph from `build_flow_runtime`. The execution-environment/IO config section is *not* called runtime — it lives at `engine.execution`. (See ADR-0006.)
_Avoid_: naming the config section `runtime`; conflating the config namespace with the live execution graph.

**Prediction**:
The canonical per-image inference result: `{labels, boxes, scores, masks?}`, where `masks` is present iff `segm` is in `data.iou_types`. Each adapter's postprocessor (head stage) is responsible for emitting exactly this shape; core code does not normalize model-specific output variants. (See ADR-0007.)
_Avoid_: per-adapter ad-hoc output shapes normalized by a core shim (`to_result_list`'s multi-branch conversion); leaking model-output variance out of the adapter.

**Postprocessor**:
The adapter-owned component (finalized in the head stage) that converts raw model outputs into canonical Predictions. It is the single place model-specific output shape is known and resolved.

**Pretrained initialization**:
Loading pretrained/backbone weights into a freshly built model during the adapter's `weights` build stage (adapter-owned, model-specific — e.g. channel remap, head differs). A *partial* load is correct by design here, because the head/channels are expected to differ from the source weights.
_Avoid_: conflating with checkpoint restore.

**Checkpoint restore**:
Loading a fully-trained framework checkpoint for eval/inference (framework-owned, generic, via `GenericCheckpointAdapter`). A *complete* load is required: a half-loaded model yields silently-wrong metrics.
_Avoid_: silently skipping missing/shape-mismatched keys on restore.

**Output base**:
The root output directory (`engine.execution`-level, CLI `--output-dir` overrides). It contains two kinds of location with opposite needs: per-run artifact dirs and shared tracking stores. (See ADR-0009.)

**Run dir**:
A per-run isolated directory `<output_base>/runs/<run_id>` holding that run's `checkpoint`, `best`, `configs`, `logs`, `inference`, and the MLflow `run_context_dir`. Runs never clobber each other. The whole tree is created in one place (`prepare_run_layout`), not scattered across callbacks.
_Avoid_: writing checkpoints/best directly into `<output_base>` (clobbers prior runs).

**Tracking store**:
A shared, run-independent location under `<output_base>` that aggregates all runs for comparison: `<output_base>/mlflow` (one MLflow store; MLflow keys each run's metrics and artifacts by its own `run_id`) and `<output_base>/tensorboard` (one TensorBoard logdir; each run is a subdir). Anchored to `<output_base>`, never to a run dir, so a single UI compares all runs. (See ADR-0009.)
_Avoid_: anchoring the MLflow tracking dir or TensorBoard logdir to the per-run dir (fragments the experiment).

**Experiment tracker**:
The component that records a run's metrics, images, text, artifacts, and execution-config provenance, via a uniform protocol with pluggable backends (MLflow, TensorBoard) composed together; a null backend covers "disabled". Renamed from "visualization logger" — visualization (`log_image`) is one method, not the purpose. The config section is `engine.execution.tracking`. MLflow is enabled and running by default and is the authoritative/persisting backend. (See ADR-0010.)
_Avoid_: "visualization logger"; treating image logging as the whole job.

**Artifact logging contract**:
`log_artifact` persists the file only on artifact-capable backends (MLflow → retrievable in its artifact store). A non-persisting backend (TensorBoard) renders a best-effort inline *preview* and is explicitly non-authoritative for retrieval. Provenance (`log_execution_config`) is part of the tracker protocol, recorded by every backend in its native form. (See ADR-0010.)
_Avoid_: assuming an artifact logged to TensorBoard is retrievable; provenance landing in MLflow only.

**Build stage**:
One of the four ordered phases an adapter runs to produce its model stack: `config` → `runtime` → `weights` → `head`. The order is fixed because it is a *data-dependency chain* (config produces `model_config`/`config_payload`; weights constructs `model_api`/`runtime_args`; head consumes them), not merely for predictability. Each stage has defined postconditions (what it must populate); `head` must end with `model`, `criterion`, `postprocessor` set or the build fails fast. One override per stage. (See ADR-0011.)
_Avoid_: reordering stages; treating the order as arbitrary; multiple overrides per stage.

**Pipeline state**:
The `AdapterPipelineState` passed through the build stages. Cross-stage handoffs are *typed fields* (`model_config`, `config_payload`, `model_api`, `runtime_args`, `model_factory`, `criterion_factory`, `model`, `criterion`, `postprocessor`); `extras` is reserved for genuinely adapter-private scratch only.
_Avoid_: using a stringly-typed `extras` dict as the inter-stage bus (invisible dependencies, KeyError surfacing stages later).
