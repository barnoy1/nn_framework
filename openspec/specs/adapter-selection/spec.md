# adapter-selection

## Purpose

Defines the stable behavior contract for adapter-selection.

## Requirements

### Requirement: Adapter selection by explicit name
The system SHALL resolve the active adapter from `adapter.name`. Substring matching against `raw_models/` directory names (`matches_source_root`) SHALL be removed.

#### Scenario: Known name resolves
- **WHEN** `adapter.name` matches exactly one registered adapter
- **THEN** the registry returns that adapter's builder

#### Scenario: Unknown name fails fast
- **WHEN** `adapter.name` matches no registered adapter
- **THEN** the system raises a hard error naming the unknown adapter

#### Scenario: Ambiguous match fails fast
- **WHEN** the config matches zero or multiple adapters
- **THEN** the system raises a hard error rather than silently picking one

#### Scenario: Renamed vendored dir does not mis-route
- **WHEN** a `raw_models/` directory is renamed but `adapter.name` is unchanged
- **THEN** selection is unaffected because it no longer depends on directory names
