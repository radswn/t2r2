from enginora.dataset.common import *


@dataclass
class TestConfig(DatasetConfigWithSelectors, WithLoadableMetrics):

    def __post_init__(self):
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [MetricsConfig(**m) for m in self.metrics]
