from dataclasses import dataclass

from t2r2.dataset.common import DatasetConfig, WithMetrics
from t2r2.utils import Stage


@dataclass
class ControlConfig(DatasetConfig, WithMetrics):
    dataset_path: str = "control.csv"
    results_file: str = "control_results.pickle"

    def __post_init__(self):
        self.stage = Stage.CONTROL
