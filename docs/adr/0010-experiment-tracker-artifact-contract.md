# ExperimentTracker with explicit per-backend artifact contract

The tracking abstraction is renamed from "visualization logger" to `ExperimentTracker`, because it records metrics, images, text, artifacts, and execution-config provenance — image logging is one method, not the purpose; the config section becomes `engine.execution.tracking`. Backends remain pluggable (MLflow, TensorBoard) behind one protocol, composed via a fan-out logger, with a null backend for "disabled".

`log_artifact` carries an explicit, non-uniform contract: it persists the file only on artifact-capable backends (MLflow, where it is retrievable in the artifact store), while a non-persisting backend such as TensorBoard renders a best-effort inline preview that is explicitly non-authoritative for retrieval. Provenance (`log_execution_config`) is promoted into the protocol so every backend records the run config in its native form, rather than living on MLflow alone.

MLflow stays enabled and running by default as the authoritative persisting backend; this change is a rename plus an honest contract, and must not break existing MLflow metric/artifact/provenance functionality.
