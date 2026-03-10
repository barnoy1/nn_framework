from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Iterable


def _sanitize_experiment_name(name: str | None, fallback: str) -> str:
    cleaned = (name or "").strip()
    return cleaned if cleaned else fallback


def _chunk_items(
    source: dict[str, str], chunk_size: int = 100
) -> Iterable[dict[str, str]]:
    items = list(source.items())
    for index in range(0, len(items), chunk_size):
        yield dict(items[index : index + chunk_size])


def build_aggregate_tracking_dir(
    root_path: Path, source_tracking_dirs: list[Path], sqlite_db_name: str
) -> Path:
    if len(source_tracking_dirs) <= 1:
        return source_tracking_dirs[0]

    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError(
            "mlflow Python package is required for multi-run aggregation mode."
        )

    import mlflow

    aggregate_dir = (root_path / ".mlflow_aggregate").resolve()
    if aggregate_dir.exists():
        shutil.rmtree(aggregate_dir)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "mlruns").mkdir(parents=True, exist_ok=True)

    destination_uri = f"sqlite:///{(aggregate_dir / sqlite_db_name).resolve()}"
    mlflow.set_tracking_uri(destination_uri)
    destination_client = mlflow.tracking.MlflowClient(tracking_uri=destination_uri)

    experiment_ids_by_name: dict[str, str] = {}

    for source_tracking_dir in source_tracking_dirs:
        source_uri = f"sqlite:///{(source_tracking_dir / sqlite_db_name).resolve()}"
        source_client = mlflow.tracking.MlflowClient(tracking_uri=source_uri)
        run_folder_name = source_tracking_dir.parent.parent.name
        experiments = source_client.search_experiments()
        for source_experiment in experiments:
            base_name = _sanitize_experiment_name(source_experiment.name, "Default")
            if base_name.lower() == "default":
                base_name = "run"
            aggregate_experiment_name = f"{base_name}_{run_folder_name}"
            destination_experiment_id = experiment_ids_by_name.get(
                aggregate_experiment_name
            )
            if destination_experiment_id is None:
                destination_experiment_id = destination_client.create_experiment(
                    aggregate_experiment_name
                )
                experiment_ids_by_name[aggregate_experiment_name] = (
                    destination_experiment_id
                )

            runs = source_client.search_runs(
                experiment_ids=[source_experiment.experiment_id],
                max_results=50000,
            )
            for source_run in runs:
                tags = dict(source_run.data.tags)
                tags["source.tracking_dir"] = str(source_tracking_dir)
                tags["source.run_id"] = source_run.info.run_id
                tags["source.artifact_uri"] = source_run.info.artifact_uri
                tags["source.run_folder"] = run_folder_name

                run_name = f"{base_name}__{run_folder_name}"
                tags["mlflow.runName"] = run_name
                destination_run = destination_client.create_run(
                    experiment_id=destination_experiment_id,
                    start_time=int(source_run.info.start_time or 0),
                    tags=tags,
                    run_name=str(run_name),
                )

                if source_run.data.params:
                    for param_chunk in _chunk_items(
                        {k: str(v) for k, v in source_run.data.params.items()}
                    ):
                        destination_client.log_batch(
                            run_id=destination_run.info.run_id,
                            params=[
                                mlflow.entities.Param(key, value)
                                for key, value in param_chunk.items()
                            ],
                        )

                if source_run.data.metrics:
                    for key, value in source_run.data.metrics.items():
                        history = source_client.get_metric_history(
                            source_run.info.run_id, key
                        )
                        if history:
                            destination_client.log_batch(
                                run_id=destination_run.info.run_id,
                                metrics=[
                                    mlflow.entities.Metric(
                                        metric.key,
                                        float(metric.value),
                                        int(metric.timestamp),
                                        int(metric.step),
                                    )
                                    for metric in history
                                ],
                            )
                        else:
                            destination_client.log_metric(
                                run_id=destination_run.info.run_id,
                                key=key,
                                value=float(value),
                                timestamp=int(
                                    source_run.info.end_time
                                    or source_run.info.start_time
                                    or 0
                                ),
                                step=0,
                            )

                destination_client.set_terminated(
                    run_id=destination_run.info.run_id,
                    status=str(source_run.info.status),
                    end_time=int(
                        source_run.info.end_time or source_run.info.start_time or 0
                    ),
                )

    return aggregate_dir
