from t2r2.dataset.common import *


@dataclass
class ControlConfig(DatasetConfig, WithMetrics):
    dataset_path: str = "control.csv"
    results_file: str = "./results/control_results.pickle"

    def __post_init__(self):
        self.stage = Stage.CONTROL
