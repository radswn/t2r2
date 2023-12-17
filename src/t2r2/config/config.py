from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml

from t2r2.dataset import ControlConfig, TestConfig, TrainingConfig, DataConfig
from t2r2.metrics import MetricsConfig
from t2r2.model import ModelConfig
from t2r2.utils.mlflow import MlFlowConfig
from t2r2.utils.repo import DvcConfig


@dataclass
class Config:
    model: ModelConfig
    metrics: List[Dict[str, Any]]
    training: TrainingConfig
    testing: TestConfig
    control: ControlConfig
    data: DataConfig
    random_state: int = None
    mlflow: MlFlowConfig = None
    dvc: DvcConfig = None

    def __post_init__(self):
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
        self.data = DataConfig(**self.data)

        self._propagate_defaults()

        self.model = ModelConfig(**self.model)
        self.training = TrainingConfig(**self.training)
        self.control = ControlConfig(**self.control)
        self.testing = TestConfig(**self.testing)
        self.mlflow = MlFlowConfig(**self.mlflow) if self.mlflow else None

        self._propagate_to_non_defaults()

        self.dvc = DvcConfig(**self.dvc) if self.dvc else DvcConfig()

    def _propagate_defaults(self):
        self._propagate_metrics()
        self._propagate_random_state()
        self._propagate_output_path()

    def _propagate_to_non_defaults(self):
        self._propagate_data_columns()
        self._propagate_dataset_path()

    def _propagate_data_columns(self):
        attributes = ["text_column_id", "label_column_id", "has_header"]
        for attr in attributes:
            for section in [self.training, self.control, self.testing]:
                if getattr(section, attr) is None:
                    setattr(section, attr, getattr(self.data, attr))

    def _propagate_metrics(self):
        self.training["metrics"] = self.metrics
        self.control["metrics"] = self.metrics
        self.testing["metrics"] = self.metrics

    def _propagate_random_state(self):
        self.training["random_state"] = self.random_state
        self.testing["random_state"] = self.random_state

        if self.mlflow:
            self.mlflow["random_state"] = self.random_state

    def _propagate_output_path(self):
        self.training["output_dir"] = self.data.output_dir
        self.control["output_dir"] = self.data.output_dir
        self.testing["output_dir"] = self.data.output_dir
        self.model["output_dir"] = self.data.output_dir

    def _propagate_dataset_path(self):
        self.training.dataset_path = self.data.data_dir + self.training.dataset_path
        self.control.dataset_path = self.data.data_dir + self.control.dataset_path
        self.testing.dataset_path = self.data.data_dir + self.testing.dataset_path


def load_config(path: str) -> Dict:
    with open(path, "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def _set_seed(random_state: int):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True


def get_config(path: str) -> Config:
    config_dict = load_config(path)
    config = Config(**config_dict)

    if config.random_state is not None:
        _set_seed(config.random_state)

    return config
