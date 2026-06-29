## ADDED Requirements

### Requirement: Flow-gated checkpoint strictness
Checkpoint load strictness SHALL depend on the flow. The train flow SHALL allow partial loads and emit a loud loaded/skipped/missing summary. The eval and inference flows SHALL be strict and fail fast on missing or shape-mismatched core weights, with an `--allow-partial` opt-out for debugging. `safe_load_state_dict` SHALL NOT silently skip mismatched weights.

#### Scenario: Train allows partial with summary
- **WHEN** the train flow loads a checkpoint missing some weights
- **THEN** the load proceeds and a loaded/skipped/missing summary is printed

#### Scenario: Eval fails fast on mismatch
- **WHEN** the eval or inference flow loads a checkpoint with missing or shape-mismatched core weights
- **THEN** the system raises unless `--allow-partial` is given

### Requirement: Head class-count compatibility check
Head class-count mismatch in the eval flow SHALL always be a hard error, regardless of `--allow-partial`. `validate_checkpoint_class_compatibility` SHALL be implemented (no longer a no-op).

#### Scenario: Class-count mismatch in eval errors
- **WHEN** eval loads a checkpoint whose head class count differs from the config
- **THEN** the system raises even if `--allow-partial` is set
