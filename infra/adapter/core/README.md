# Simple Adapter Quickstart

This adapter flow is intentionally explicit and easy to extend.

## Goal

Add a new model adapter without touching reflection internals.

## Steps

1. Create a builder class in your adapter package (example: `infra/adapter/my_model/model_builder.py`).
2. Make that builder implement `ModelBuilder` by returning `BuiltComponents` through the existing runtime path.
3. Register it once in `infra/adapter/core/registry.py` by adding one `AdapterSpec` entry.

```python
AdapterSpec(
    name="my_model",
    source_root_tokens=("my-model", "my_model"),
    builder_factory=MyModelBuilder,
)
```

That is all. Runtime selection is automatic through `resolve_model_builder(...)`.

## Built-in adapter structure

Both built-in adapters follow the same package shape:

- `infra/adapter/rf_detr/runtime/`
- `infra/adapter/rtdetrv2_pytorch/runtime/`

Each runtime package uses the same files:

- `config.py`
- `backbone.py`
- `variant.py`
- `weights.py`
- `__init__.py`

Put adapter-specific logic in `runtime/` and keep `model_builder.py` as a thin orchestrator.

## Why this is simpler

- One registration point (`registry.py`)
- One matching rule (`source_root_tokens`)
- One builder entrypoint (`builder_factory(app_config, repo_root)`)

No YAML reflection knowledge is required to add a basic adapter.