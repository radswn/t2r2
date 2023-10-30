from typing import Tuple

from sklearn.model_selection import train_test_split
from transformers import IntervalStrategy, TrainingArguments

from enginora.dataset.common import *


@dataclass
class TrainingConfig(DatasetConfigWithSelectors, WithMetrics):
    dataset_path: str = "./data/train.csv"
    text_column_id: int = 0
    label_column_id: int = 1
    has_header: bool = True
    output_dir: str = "./results/"
    results_file: str = "./results/train_results.pickle"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    validation_size: float = 0.2
    metric_for_best_model: str = "accuracy_score"

    def __post_init__(self):
        self.stage = Stage.TRAINING
        self.selectors = [] if self.selectors is None else [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)
        self.validation_size = float(self.validation_size)

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = super().load_dataset()
        X = data["text"]
        y = data["label"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_size, random_state=self.random_state
        )
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        return train_data, val_data

    def get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            logging_strategy=IntervalStrategy.EPOCH,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_for_best_model,
            num_train_epochs=self.epochs,
            report_to=None,
        )
