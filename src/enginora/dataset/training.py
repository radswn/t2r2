from enginora.dataset.common import *


@dataclass
class TrainingValidationConfig(DatasetConfigComplex):
    training_path: str
    batch_size: int
    epochs: int
    learning_rate: float
    output_dir: str

    validation_path: str
    metric_for_best_model: str

    def __post_init__(self):
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)

        self.metrics = [MetricsConfig(**m) for m in self.metrics]
