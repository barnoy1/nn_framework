# 03. Dummy Adapter Step-by-Step

This section explains the full tutorial adapter in `infra/adapter/tutorial_dummy_unet/`.

## Full structure

```text
infra/adapter/tutorial_dummy_unet/
  __init__.py
  model_builder.py
  manifest.py
  schemes.py
  runtime/
    __init__.py
    config.py
    variant.py
    weights.py
  overrides/
    __init__.py
    config_overrides.py
    runtime_overrides.py
    weight_overrides.py
    head_overrides.py
```

## Step 1: builder shell (`model_builder.py`)

Purpose: orchestration only.

Responsibilities:

1. inherit `StagedAdapterModelBuilder`
2. pass `adapter_root`
3. expose `manifest()` classmethod

No model-specific runtime logic should live here.

## Step 2: manifest (`manifest.py`)

Purpose: declarative adapter contract.

Defines:

- `name`
- `source_root_aliases`
- `builder_factory`
- `config_subdir`
- `overrides_by_stage`

The staged order is fixed by core spec and validated on load.

### Manifest field deep dive

#### `source_root_aliases`

What it is: matching aliases used by `resolve_model_builder(...)` against `app_config.model.source_root`.

Why needed: runtime flow remains generic; adapter choice becomes data-driven.

Best practice: include lowercase variants and common aliases.

#### `builder_factory`

What it is: builder class constructor (`builder_factory(app_config, repo_root)`).

Why needed: registry builds concrete builders without hardcoding internals.

Best practice: return the same class from `model_builder.py` that exposes `manifest()`.

#### `overrides_by_stage`

What it is: mapping of stage name to ordered override tuple.

Why needed: explicit, deterministic behavior injection points.

Best practice: always define all four keys (`config`, `runtime`, `weights`, `head`) even if some are pass-through.

#### `config_subdir`

What it is: fallback location for model config files relative to adapter root.

Why needed: allows adapter-local configs when app config path is relative.

Best practice: keep adapter-owned example configs under this folder.

## Step 3: runtime helpers (`runtime/*`)

Purpose: small reusable operations for stages.

### `runtime/config.py`

- import-path preparation
- YAML payload loading

### `runtime/variant.py`

- build `DummyUNetRuntimeAPI` from payload

### `runtime/weights.py`

- optional checkpoint loading policy

## Step 4: stage overrides (`overrides/*`)

Each override has one reason to change.

### `config_overrides.py`

- ensure paths
- load payload into `state.config_payload`
- apply single-channel policy (`in_channels: 3 -> 1`)

### `runtime_overrides.py`

- build concrete runtime API
- store in `state.model_api`

### `weight_overrides.py`

- apply optional checkpoint path from payload
- adapt first conv checkpoint weights from 3-channel to 1-channel when needed

### `head_overrides.py`

- set required framework outputs:
  - `state.model`
  - `state.criterion`
  - `state.postprocessor`

## Step 5: (optional) registration

This tutorial adapter is intentionally not auto-registered.

To activate it, add its builder in `infra/adapter/core/registry.py` the same way production adapters are included.

## Exact 3ch -> 1ch walkthrough (where it happens)

Follow this exact path:

1. Start config with RGB declaration:
   - `infra/adapter/tutorial_dummy_unet/configs/tutorial_dummy_unet.yaml`
   - `model.in_channels: 3`
2. Config stage loads payload:
   - `runtime/config.py::load_tutorial_payload(...)`
3. Adapter policy mutates payload:
   - `runtime/config.py::apply_single_channel_adapter_policy(...)`
   - sets:
     - `requested_in_channels=3`
     - `in_channels=1`
     - `adapter_channel_policy="force-single-channel"`
4. Runtime stage builds concrete model from mutated payload:
   - `runtime/variant.py::build_runtime_api(...)`
   - `raw_models/dummy_unet/src/api.py::DummyUNetRuntimeAPI.from_payload(...)`
5. Concrete model is instantiated with `in_channels=1`:
   - `raw_models/dummy_unet/src/simple_unet.py::SimpleUNet(...)`
6. Weight stage adapts checkpoint first conv when 3ch checkpoint is loaded into 1ch model:
   - `runtime/weights.py::_adapt_first_conv_to_single_channel(...)`
   - conversion: mean over checkpoint channel dim (3 -> 1)

## Run the walkthrough script

```bash
python infra/adapter/tutorial_dummy_unet/main.dummy.py \
  --config infra/adapter/tutorial_dummy_unet/configs/tutorial_dummy_unet.yaml
```

Expected output includes:

- `config requested in_channels=3`
- `adapter effective in_channels=1`
- first conv shape showing 1 input channel
- metadata with `adapter_channel_policy=force-single-channel`

## What users should copy into real integrations

1. staged override layout
2. thin builder + declarative manifest
3. concrete runtime API boundary
4. explicit source-root matching tokens
5. strict `head` stage completion
