from enginora.dataset.common import *


@dataclass
class TrainingConfig(DatasetConfigWithSelectors):
    batch_size: int
    epochs: int
    learning_rate: float
    output_dir: str

    def __post_init__(self):
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)


@dataclass
class ValidationConfig(DatasetConfig):
    # TODO: merge with training set
    batch_size: int
    metrics: List[MetricsConfig]
    metric_for_best_model: str

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.metrics = [MetricsConfig(**m) for m in self.metrics]

    def compute_metrics(self, predictions) -> dict:
        predictions, true_labels = predictions[0], predictions[1]
        predictions = predictions.argmax(1)

        return {
            metric.name:
                get_metric(metric.name)(true_labels, predictions)
            for metric in self.metrics
        }
