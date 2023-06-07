from enginora.dataset.common import *


@dataclass
class ControlConfig(DatasetConfig, WithLoadableMetrics):
    def __post_init__(self):
        self.metrics = [MetricsConfig(**m) for m in self.metrics]
