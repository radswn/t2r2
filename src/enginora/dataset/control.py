from enginora.dataset.common import *
from dataclasses import dataclass, field


@dataclass
class ControlConfig(DatasetConfig, WithLoadableMetrics):
    def __post_init__(self):
        self.stage = Stage.CONTROL
        self.metrics = [MetricsConfig(**m) for m in self.metrics]
