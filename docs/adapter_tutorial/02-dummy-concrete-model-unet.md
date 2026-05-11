# 02. Dummy Concrete Model (Simple UNet)

This section documents the tutorial concrete model placed in `raw_models/dummy_unet/`.

## Why add a concrete model example

Adapter tutorials are hard to follow when they only show framework-side glue code.

A real model-side implementation helps users understand:

1. what the adapter expects from model runtime APIs
2. how model, criterion, and postprocessor are exposed
3. where optional checkpoint loading hooks belong

## Files

- `raw_models/dummy_unet/src/simple_unet.py`
- `raw_models/dummy_unet/src/api.py`

## What the model represents

It is a lightweight segmentation-style architecture that mimics UNet encoder/decoder behavior:

- encoder blocks + pooling
- bottleneck block
- decoder blocks + skip concatenation
- final 1x1 prediction head

It is intentionally minimal and educational, not benchmark-oriented.

## Runtime API contract exposed

`api.py` provides:

- `DummyUNetConfig`: typed config for channels/classes.
- `DummyUNetRuntimeAPI`: container exposing:
  - `model`
  - `criterion`
  - `postprocessor`

This shape mirrors what adapter `head` stage must finalize for the framework.

## Criterion and postprocessor choices

The tutorial uses:

- `DummySegCriterion`: `CrossEntropyLoss` over logits/masks.
- `DummySegPostprocessor`: argmax-based label mask projection.

This demonstrates how model-side APIs can hide internal task details while presenting a stable interface to the adapter.

## Channel mismatch tutorial scenario

The concrete model default is **3 input channels** (`in_channels=3`).

In the tutorial adapter, we intentionally force a **1-channel runtime policy** to demonstrate a common real-world case:

1. model repo defaults to RGB
2. deployment data is grayscale
3. adapter applies channel policy without modifying framework flows

The policy is implemented in adapter code, not in train/eval/inference managers:

- config stage forces `in_channels=1`
- weights stage adapts first conv weights from 3-channel checkpoint to 1-channel (channel mean)

## Why this structure matters for users

When integrating a real model repo, users can mirror the same concept:

1. keep model architecture in one module
2. keep runtime API composition (model + criterion + postprocessor) in one module
3. keep framework adapter thin and staged

That separation dramatically reduces coupling between framework internals and concrete model internals.
