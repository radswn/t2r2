from enginora.dataset.common import *
from typing import List


@dataclass
class TrainingValidationConfig(DatasetConfigComplex):
    dataset_path: List
    batch_size: int
    epochs: int
    learning_rate: float
    output_dir: str

    metric_for_best_model: str

    def __post_init__(self):
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)
        self.training_path = self.dataset_path[0]
        self.validation_path = self.dataset_path[1]

        self.metrics = [MetricsConfig(**m) for m in self.metrics]