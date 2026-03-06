from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CommonConfig(BaseModel):
    output_dir: str = "output"
    checkpoint: str = ""
    device: str = "cuda"
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    use_gpu: bool = True
    use_xpu: bool = False
    use_mlu: bool = False
    use_npu: bool = False
    log_iter: int = 20
    epoches: Optional[int] = None
    score_threshold: float = 0.3

    @field_validator("epoches")
    @classmethod
    def validate_epoches(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("runtime.common.epoches must be > 0 when provided")
        return value

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, value: float) -> float:
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("runtime.common.score_threshold must be in [0, 1]")
        return numeric

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and int(value) <= 0:
            raise ValueError("runtime.common.batch_size must be > 0 when provided")
        return value

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and int(value) < 0:
            raise ValueError("runtime.common.num_workers must be >= 0 when provided")
        return value


class DataPreparationConfig(BaseModel):
    prepare_data: bool = False
    supervisely_dataset_root: Optional[str] = None
    supervisely_splits: List[str] = Field(default_factory=lambda: ["train", "valid"])
    ann_subdir: str = "ann"
    img_subdir: str = "img"
    mlflow_experiment_name: Optional[str] = None


class ExportConfig(BaseModel):
    post_process: bool = True
    nms: bool = True
    benchmark: bool = False
    fuse_conv_bn: bool = False


class TensorBoardVisualizationConfig(BaseModel):
    enabled: bool = False
    log_dir: str = "runs/visualization"
    host: str = "127.0.0.1"
    port: int = 6006
    start_service: bool = True


class MlflowVisualizationConfig(BaseModel):
    enabled: bool = False
    mlflow_dir: str = "mlflow"
    tracking_backend: str = "sqlite"
    sqlite_db_name: str = "mlflow.db"
    host: str = "127.0.0.1"
    port: int = 5000
    start_service: bool = True


class VisualizationConfig(BaseModel):
    num_samples: int = 16
    tensorboard: TensorBoardVisualizationConfig = Field(default_factory=TensorBoardVisualizationConfig)
    mlflow: MlflowVisualizationConfig = Field(default_factory=MlflowVisualizationConfig)


class RuntimeConfig(BaseModel):
    description: Optional[str] = None
    common: CommonConfig = Field(default_factory=CommonConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    data_preparation: DataPreparationConfig = Field(default_factory=DataPreparationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    actions: Dict[str, Dict] = Field(default_factory=dict)

    @property
    def use_gpu(self) -> bool:
        return self.common.use_gpu

    @property
    def use_xpu(self) -> bool:
        return self.common.use_xpu

    @property
    def use_mlu(self) -> bool:
        return self.common.use_mlu

    @property
    def use_npu(self) -> bool:
        return self.common.use_npu

    @property
    def log_iter(self) -> int:
        return self.common.log_iter

    @property
    def output_dir(self) -> str:
        return self.common.output_dir

    @property
    def batch_size(self) -> Optional[int]:
        return self.common.batch_size

    @property
    def num_workers(self) -> Optional[int]:
        return self.common.num_workers

    @property
    def epoches(self) -> Optional[int]:
        return self.common.epoches

    @property
    def prepare_data(self) -> bool:
        return self.data_preparation.prepare_data

    @property
    def supervisely_dataset_root(self) -> Optional[str]:
        return self.data_preparation.supervisely_dataset_root

    @property
    def supervisely_splits(self) -> List[str]:
        return self.data_preparation.supervisely_splits

    @property
    def ann_subdir(self) -> str:
        return self.data_preparation.ann_subdir

    @property
    def img_subdir(self) -> str:
        return self.data_preparation.img_subdir

    @property
    def mlflow_experiment_name(self) -> Optional[str]:
        return self.data_preparation.mlflow_experiment_name
