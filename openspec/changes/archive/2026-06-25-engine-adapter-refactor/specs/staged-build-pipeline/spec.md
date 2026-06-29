## ADDED Requirements

### Requirement: Typed inter-stage build contract
`AdapterPipelineState` SHALL expose inter-stage handoffs as typed fields: `model_config`, `config_payload`, `model_api`, `runtime_args`, `model_factory`, `criterion_factory`, `model`, `criterion`, `postprocessor`. The untyped `extras` bag SHALL be demoted to private scratch and SHALL NOT carry inter-stage handoffs.

#### Scenario: Missing handoff fails at definition
- **WHEN** a stage fails to set a required typed field its successor needs
- **THEN** the error surfaces at the stage boundary, not stages later

#### Scenario: extras is private scratch
- **WHEN** a downstream stage attempts to read an inter-stage value from `extras`
- **THEN** no such contract exists because handoffs are typed fields

### Requirement: Enforced per-stage postconditions
Each build stage SHALL enforce its postconditions. The head stage SHALL end with `model`, `criterion`, and `postprocessor` all set. Each stage SHALL have exactly one override (`overrides_by_stage` collapsed to one override per stage).

#### Scenario: Head postcondition enforced
- **WHEN** the head stage completes without setting `postprocessor`
- **THEN** the build raises before returning `BuiltComponents`

#### Scenario: One override per stage
- **WHEN** an adapter supplies stage overrides
- **THEN** at most one override per stage is accepted
