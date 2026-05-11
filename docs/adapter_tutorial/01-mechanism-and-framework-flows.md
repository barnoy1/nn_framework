# 01. Mechanism and Framework Main Flows

This section explains how adapters connect model-specific code to framework-level flows.

## Core idea

The framework keeps train/eval/inference managers generic. Model-specific behavior is isolated inside adapters that satisfy one shared contract.

The handoff chain is:

1. Flow manager (`train`, `eval`, `inference`) calls `build_flow_runtime(...)`.
2. Runtime builder creates model wrapper through `create_wrapper(...)`.
3. Wrapper asks adapter registry to resolve a model builder based on `app_config.model.source_root`.
4. Resolved builder constructs `BuiltComponents` (model, criterion, postprocessor, optimizer, scheduler, etc.).
5. Flow manager uses those components without knowing model internals.

## Where this happens in code

- Runtime composition: `infra/engine/flows/common/runtime.py`
- Wrapper factory: `infra/engine/model/wrappers/component_factory.py`
- Adapter selection: `infra/adapter/core/registry.py`
- Wrapper runtime facade: `infra/engine/model/wrappers/adapter_runtime.py`
- Tutorial code launcher: `infra/adapter/tutorial_dummy_unet/main.dummy.py`

## Why adapter abstraction is needed

Different model repos differ in:

- config format (YAML/JSON/custom objects)
- weight-loading semantics
- postprocessing logic
- criterion construction
- optional runtime patches

Without adapters, train/eval/inference flows would embed model-specific conditionals and become brittle.

Adapters isolate those differences behind one common return type (`BuiltComponents`).

## Staged adapter mechanism

The staged builder (`StagedAdapterModelBuilder`) executes deterministic phases:

1. `config`
2. `runtime`
3. `weights`
4. `head`

Each phase updates a shared `AdapterPipelineState`.

### Why staged phases

- **Predictability**: all adapters run the same lifecycle.
- **Auditability**: easier to inspect where behavior is injected.
- **Extensibility**: add model-specific logic in one explicit stage, not in ad-hoc builder code.

## What each stage represents

### `config`

Purpose: resolve and normalize model config inputs.

Typical tasks:

- ensure import paths
- load config payload
- map aliases/variants
- prepare model config objects

### `runtime`

Purpose: bind concrete runtime constructors and APIs.

Typical tasks:

- choose model API class/factory
- prepare intermediate runtime objects
- attach stage-local helpers in `state.extras`

### `weights`

Purpose: apply pretrained/checkpoint policy.

Typical tasks:

- resolve download path
- partial state dict load
- optional channel-mismatch policy
- initialize runtime args from constructed model API

### `head`

Purpose: finalize components for engine consumption.

Required outputs:

- `state.model`
- `state.criterion`
- `state.postprocessor`

If these are not set, staged builder fails fast.

## How this links to main flows

Train, eval, and inference all use the same runtime builder and wrapper resolution path.

That means:

- adding a new adapter can enable all three flows at once
- callers still run the same CLI/API commands
- model-specific complexity is fully internal to adapter/runtime packages

## Invariants you should preserve

1. Do not change caller-side flow APIs.
2. Keep adapter selection based on `source_root_aliases`.
3. Ensure deterministic override order.
4. Keep builder orchestration-only; move specifics to staged overrides.
5. Return fully formed `BuiltComponents` to the wrapper layer.
