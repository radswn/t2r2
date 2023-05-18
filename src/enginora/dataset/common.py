import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd

from enginora.metrics import MetricsConfig, get_metric
from enginora.selector import get_selector, SelectorConfig


@dataclass
class DatasetConfig:
    dataset_path: str

    def load_dataset(self) -> pd.DataFrame:
        # TODO: what columns should be accept as text/target (?)
        return pd.read_csv(self.dataset_path, header=None, names=['id', 'text', 'label'])


@dataclass
class DatasetConfigWithSelectors(DatasetConfig):
    selectors: List[SelectorConfig]

    def load_dataset(self) -> pd.DataFrame:
        df = super().load_dataset()

        for t in self.selectors:
            selector = get_selector(t.name)(**t.args)
            df = selector.select(df)

        # TODO: save the dataset after selectors
        return df


@dataclass
class DatasetConfigWithMetrics(DatasetConfig):
    metrics: List[MetricsConfig]
    results_file: str

    def save_predictions(self, predictions):
        with open(self.results_file, 'wb') as file:
            pickle.dump(predictions, file)

    def load_predictions(self):
        with open(self.results_file, 'rb') as file:
            return pickle.load(file)

    def compute_metrics(self) -> dict:
        predictions = self.load_predictions()
        predictions, true_labels = predictions[0], predictions[1]
        predictions = predictions.argmax(1)

        return {
            metric.name:
                get_metric(metric.name)(true_labels, predictions)
            for metric in self.metrics
        }
