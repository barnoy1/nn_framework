# Adapter Manifest Quickstart

This adapter flow is manifest-driven, explicit, and easy to extend.

## Goal

Add a new model adapter without changing runner flow or engine wrapper contracts.

## Steps

1. Create `manifest.py` in your adapter package and return `AdapterManifest`.
2. Implement stage overrides under `overrides/` (`config`, `runtime`, `weights`, `head`).
3. Keep `model_builder.py` orchestration-only by inheriting the staged builder base.
4. Expose `manifest()` from the builder class. `core/registry.py` auto-loads it.

Runtime selection remains automatic through `resolve_model_builder(...)`.

## Built-in adapter structure

Both built-in adapters follow the same package shape:

- `infra/adapter/<name>/manifest.py`
- `infra/adapter/<name>/model_builder.py`
- `infra/adapter/<name>/overrides/`
- `infra/adapter/<name>/runtime/`

`model_builder.py` should not contain adapter-specific patch logic. Keep logic in override modules and `runtime/`.

## Why this is simpler

- One typed manifest per adapter
- One deterministic override order
- One builder entrypoint (`builder_factory(app_config, repo_root)`)

No YAML reflection knowledge is required to add a basic adapter.
