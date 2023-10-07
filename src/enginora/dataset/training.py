from typing import List, Tuple

from sklearn.model_selection import train_test_split
from transformers import TrainerCallback

from enginora.dataset.common import *
from enginora.utils.data_cartography.data_cartography import compute_data_cartography_metrics, create_plot
from enginora.utils.save_predictions_callback import SavePredictionsCallback

@dataclass
class TrainingConfig(DatasetConfigWithSelectors, WithMetrics):
    dataset_path: str = "./data/train.csv"
    validation_dataset_path: str = None
    output_dir: str = "./results/"
    results_file: str = "./results/train_results.pickle"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    validation_size: float = 0.2
    metric_for_best_model: str = "accuracy_score"
    perform_data_cartography: bool = False
    data_cartography_results: str = './data_cartography_metrics.pickle'
    callbacks: Tuple[TrainerCallback, ...] = ()

    def load_validation_dataset(self) -> pd.DataFrame:
        """ Method for loading validation dataset if one exists"""
        # TODO: what columns should be accept as text/target (?)
        # TODO: maybe create ids if not provided (?)
        return pd.read_csv(self.validation_dataset_path, header=None, names=["id", "text", "label"])

    def __post_init__(self):
        self.stage = Stage.TRAINING
        self.selectors = [] if self.selectors is None else [SelectorConfig(**t) for t in self.selectors]
        self.metrics = [] if self.metrics is None else [MetricsConfig(**m) for m in self.metrics]
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.learning_rate = float(self.learning_rate)
        self.validation_size = float(self.validation_size)
        self.callbacks = (SavePredictionsCallback(),) if self.perform_data_cartography else ()

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = super().load_dataset()
        X = data[["id", "text"]]
        y = data["label"]

        if self.validation_dataset_path is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_size, random_state=self.random_state
            )
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
        else:
            val_data = self.load_validation_dataset()
            X_val = val_data[["id", "text"]]
            y_val = val_data["label"]
            train_data = pd.concat([X, y], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
        return train_data, val_data

    def get_callbacks(self) -> List[TrainerCallback]:
        """Gets Callback which are passed to Trainer"""
        return list(self.callbacks)

    def save_results(self, results, mlflow_manager: MlflowManager):
        super().save_results(results, mlflow_manager)

        if self.perform_data_cartography:
            self.data_cartography()

    def data_cartography(self):
        save_prediction_callback = next((callback for callback in self.get_callbacks() if isinstance(callback, SavePredictionsCallback)), None)
        predictions = pd.DataFrame(save_prediction_callback.get_predictions())
        labels = save_prediction_callback.get_labels()
        number_of_epochs = save_prediction_callback.get_epochs()
        df_metrics_for_data_cartography = compute_data_cartography_metrics(predictions, labels, number_of_epochs)

        with open(self.data_cartography_results, "wb") as file:
            pickle.dump(df_metrics_for_data_cartography, file)

        create_plot(df_metrics_for_data_cartography, self.output_dir)
