from typing import Tuple

from sklearn.model_selection import train_test_split

from enginora.dataset.common import *
from dataclasses import field

@dataclass
class TrainingConfig(DatasetConfigWithSelectors, WithMetrics):
    batch_size: int
    epochs: int
    learning_rate: float
    validation_size: float
    output_dir: str
    metric_for_best_model: str

    def __post_init__(self):
        self.stage = Stage.TRAINING
        self.selectors = [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [MetricsConfig(**m) for m in self.metrics]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)
        self.validation_size = float(self.validation_size)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = super().load_dataset()
        X = data[["id", "text"]]
        y = data["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size
        )
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        return train_data, val_data
