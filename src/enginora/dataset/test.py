from enginora.dataset.common import *
from dataclasses import field


@dataclass
class TestConfig(DatasetConfigWithSelectors, WithLoadableMetrics):
    def __post_init__(self):
        self.stage = Stage.TESTING
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [MetricsConfig(**m) for m in self.metrics]
