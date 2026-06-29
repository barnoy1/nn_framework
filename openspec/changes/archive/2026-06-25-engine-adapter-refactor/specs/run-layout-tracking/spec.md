## ADDED Requirements

### Requirement: Per-run isolated artifact directories
Heavy artifacts SHALL be written under `<output_base>/runs/<run_id>/{checkpoint,best,configs,logs,inference}`. The whole directory tree SHALL be created once in `prepare_run_layout`. Two consecutive runs SHALL keep independent `runs/<run_id>/checkpoint` directories.

#### Scenario: Consecutive runs do not clobber
- **WHEN** two runs execute back to back
- **THEN** each writes to its own `runs/<run_id>/checkpoint` without overwriting the other

#### Scenario: Layout created once
- **WHEN** a run starts
- **THEN** `prepare_run_layout` creates the full per-run tree in one step

### Requirement: Shared tracking stores
MLflow SHALL use one shared store at `<output_base>/mlflow` keyed by MLflow `run_id`. TensorBoard SHALL write per-run logs under `<output_base>/tensorboard/<run_id>`. MLflow SHALL remain default-on and authoritative; metric keys and artifacts SHALL be structurally identical to `main`.

#### Scenario: One store, many runs
- **WHEN** multiple runs complete
- **THEN** all MLflow runs live in one shared store and one UI compares them

#### Scenario: Per-run TB subdir
- **WHEN** a run logs scalars
- **THEN** they are written under `tensorboard/<run_id>` distinct from other runs

#### Scenario: Tracking parity with main
- **WHEN** a 2-epoch smoke run completes
- **THEN** its MLflow metric-key set and retrievable artifacts match `main` in structure

### Requirement: ExperimentTracker contract
The "visualization logger" SHALL be renamed `ExperimentTracker`, configured via `engine.execution.tracking`. `log_artifact` SHALL persist on MLflow and produce a best-effort, non-authoritative preview on TensorBoard. `log_execution_config` SHALL be part of the protocol.

#### Scenario: Artifact persistence vs preview
- **WHEN** `log_artifact` is called
- **THEN** MLflow persists it authoritatively and TensorBoard gets a best-effort preview

#### Scenario: Execution config logged
- **WHEN** a run starts
- **THEN** `log_execution_config` records the resolved execution config through the tracker protocol
