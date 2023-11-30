from dataclasses import dataclass

from t2r2.dataset.common import DatasetConfigWithSelectors, WithMetrics
from t2r2.selector import SelectorConfig
from t2r2.utils import Stage


@dataclass
class TestConfig(DatasetConfigWithSelectors, WithMetrics):
    dataset_path: str = "test.csv"
    results_file: str = "./results/test_results.pickle"

    def __post_init__(self):
        self.stage = Stage.TESTING
        self.selectors = [] if self.selectors is None else [SelectorConfig(**t) for t in self.selectors]
