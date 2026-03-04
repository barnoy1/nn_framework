# Engine Refactor Plan (SOLID + Clean Code)

## Scope

This plan covers only `infra/engine` and follows a non-breaking migration strategy:
- preserve existing entrypoints (`mangr_*`),
- preserve current public imports where already used,
- avoid placeholder/empty files,
- split by responsibility with immediate runtime usage.

## Hotspots Found

1. `trainer.py` (~339 LOC):
   - mixes lifecycle orchestration, loss-component parsing, validation loss computation, train-batch visualization, EMA handling.
2. `model/factory.py` (~79 LOC):
   - mixes dynamic module loading + wrapper creation + checkpoint/dn factories.
3. Existing flow modules now improved, but trainer internals still tightly coupled.

## Target Architecture

### A. Trainer internals split into training helpers

Create `infra/engine/training/` package:
- `loss_components.py`
  - loss-term matching and component split/aggregation.
- `validation.py`
  - validation loss pass and EMA context management.
- `visualization.py`
  - train-batch image panel rendering.

`trainer.py` remains orchestration-focused and delegates helper concerns.

### B. Model factory decomposition

Split dynamic loading from creation:
- `model/module_loader.py`
  - wrapper module loading by path.
- `model/wrapper_creators.py`
  - wrapper/builder/checkpoint/dn creator functions.

Keep `model/factory.py` as compatibility facade re-exporting the same public functions.

## Phases

### Phase 1 (execute now)
1. Add training helper package and wire `trainer.py` to it.
2. Keep `Trainer` API and behavior unchanged.

### Phase 2 (execute now)
1. Add `module_loader.py` + `wrapper_creators.py`.
2. Convert `factory.py` into compatibility facade.
3. Keep `infra.engine.model` public API stable.

### Phase 3 (validate)
1. Static checks on modified files.
2. Import smoke tests for trainer/model factories.
3. Flow help smoke tests (`train/eval/inference`).

## Validation Criteria

- No import regressions in `infra.engine`.
- `build_flow_runtime(...)` and flow managers still load.
- `Trainer.fit()` contract unchanged from caller perspective.
- No new placeholder files.
