from enginora.dataset.common import *


@dataclass
class ControlConfig(DatasetConfig, WithMetrics):
    dataset_path: str = "./data/control.csv"
    results_file: str = "./results/control_results.pickle"

    def __post_init__(self):
        self.stage = Stage.CONTROL
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
