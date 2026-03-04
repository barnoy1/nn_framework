from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CommonConfig(BaseModel):
    model_profile: str = "r18"
    output_dir: str = "output"
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 2
    use_gpu: bool = True
    use_xpu: bool = False
    use_mlu: bool = False
    use_npu: bool = False
    log_iter: int = 20
    snapshot_epoch: int = 1
    print_flops: bool = False
    print_params: bool = False
    epoches: Optional[int] = None

    @field_validator("epoches")
    @classmethod
    def validate_epoches(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("runtime.common.epoches must be > 0 when provided")
        return value


class DataPreparationConfig(BaseModel):
    prepare_data: bool = False
    supervisely_dataset_root: Optional[str] = None
    supervisely_splits: List[str] = Field(default_factory=lambda: ["train", "valid"])
    ann_subdir: str = "ann"
    img_subdir: str = "img"
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None


class ExportConfig(BaseModel):
    post_process: bool = True
    nms: bool = True
    benchmark: bool = False
    fuse_conv_bn: bool = False


class TensorBoardVisualizationConfig(BaseModel):
    enabled: bool = False
    log_dir: str = "runs/visualization"


class MlflowVisualizationConfig(BaseModel):
    enabled: bool = False
    mlflow_dir: str = "mlflow"


class VisualizationConfig(BaseModel):
    num_samples: int = 16
    tensorboard: TensorBoardVisualizationConfig = Field(default_factory=TensorBoardVisualizationConfig)
    mlflow: MlflowVisualizationConfig = Field(default_factory=MlflowVisualizationConfig)


class RuntimeConfig(BaseModel):
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
    def snapshot_epoch(self) -> int:
        return self.common.snapshot_epoch

    @property
    def print_flops(self) -> bool:
        return self.common.print_flops

    @property
    def print_params(self) -> bool:
        return self.common.print_params

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

    @property
    def mlflow_run_name(self) -> Optional[str]:
        return self.data_preparation.mlflow_run_name
