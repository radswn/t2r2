from t2r2.dataset.common import *


@dataclass
class ControlConfig(DatasetConfig, WithMetrics):
    dataset_path: str = "./data/control.csv"
    text_column_id: int = 0
    label_column_id: int = 1
    has_header: bool = True
    results_file: str = "./results/control_results.pickle"

    def __post_init__(self):
        self.stage = Stage.CONTROL
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
