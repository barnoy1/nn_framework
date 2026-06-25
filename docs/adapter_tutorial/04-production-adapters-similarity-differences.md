# 04. Production Adapters: Similarities and Differences

This section compares `rf_detr` and `rtdetrv2_pytorch` using the staged model.

## Shared structure (similarities)

Both adapters now follow:

1. thin `model_builder.py`
2. manifest-driven contract
3. four explicit stages
4. runtime helper modules
5. identical wrapper/flow integration path

Both are selected by their manifest `name` (matched against `adapter.name`) and resolved via `resolve_model_builder(...)`.

## Behavior-level differences

| Area | RF-DETR | RT-DETRv2 |
|---|---|---|
| Config stage | DINO config load + profile + channel policy | Import-path and config setup |
| Runtime stage | Binds RF-DETR model/criterion factories | Applies runtime backbone policy |
| Weights stage | Pretrained path resolution and partial loading logic | Weight policy application on runtime config |
| Head stage | Head realignment + criterion build from runtime args | Direct component load from YAMLConfig |

## Why both still fit one mechanism

The stage names are semantic boundaries, not strict algorithm definitions.

So one adapter may use a stage as a pass-through while another performs substantial logic in the same stage. That is expected and still valuable because the execution lifecycle remains consistent.

## Practical design lessons

1. Keep high-variance logic in stage files, not in builders.
2. Normalize legacy aliases in dedicated compatibility helpers.
3. Treat config/weights/head operations as separate concerns.
4. Preserve the framework-facing contract regardless of model internals.

## Mapping from tutorial dummy adapter to production adapters

The dummy adapter demonstrates exactly the same shape:

- manifest contract
- staged overrides
- runtime API handoff
- final components contract

Users can start from dummy files, then incrementally replace runtime internals with real model-repo logic.

## Extra tutorial pattern: channel-policy adaptation

The dummy adapter includes a practical pattern not shown as explicitly in this comparison:

1. concrete model default is 3-channel
2. adapter enforces 1-channel runtime policy
3. checkpoint loading adapts first-conv weights (3 -> 1 by mean)

This mirrors real adapter responsibilities where data modality differs from model defaults.
