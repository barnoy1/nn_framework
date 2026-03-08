# Wrapper Integration Guide

This guide describes how to integrate a new model backend into the framework using the wrapper contract.

## Contract Location

Framework-side contract:
- `infra/engine/model/wrappers/adapter_api.py`

Required public function in concrete wrapper module (`nn_wrapper/adapter.py`):
- `create_wrapper_components(app_config, repo_root) -> WrapperComponents`

Optional compatibility function:
- `create_model_builder(app_config, repo_root) -> ModelBuilder`

## Minimal Concrete Wrapper

Your concrete wrapper should expose `create_wrapper_components(...)` and return a `WrapperComponents` with a model builder:

- `model_builder`: instance implementing `ModelBuilder.build()`

The framework owns:
- checkpoint loading and safe state-dict application
- wrapper runtime adapter construction

## Builder Responsibilities

Concrete `ModelBuilder` should implement:
- `build() -> BuiltComponents`
- `apply_architecture_specifics(model, targets, *, dn_num_group: int) -> None` (only if your architecture needs extra runtime adjustments)

`build()` is expected to return:
- model
- criterion
- postprocessor
- optimizer
- scheduler
- optional EMA model
- optional class-id mapping

## Loading Flow

Framework loader (`component_factory.py`) does the following:
1. Loads `nn_wrapper/adapter.py`
2. Validates required contract function(s)
3. Calls `create_wrapper_components(...)`
4. Builds `FrameworkModelAdapter` around returned builder

## Recommendations

- Keep concrete wrappers model-specific only.
- Put framework-agnostic logic in `infra/engine/model/wrappers/common`.
- Avoid exposing extra public functions unless required by the contract.
