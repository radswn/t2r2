from t2r2.dataset.common import *


@dataclass
class TestConfig(DatasetConfigWithSelectors, WithMetrics):
    dataset_path: str = "./data/test.csv"
    text_column_id: int = 0
    label_column_id: int = 1
    has_header: bool = True
    results_file: str = "test_results.pickle"

    def __post_init__(self):
        self.stage = Stage.TESTING
        self.selectors = [] if self.selectors is None else [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
