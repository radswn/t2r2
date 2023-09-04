import os.path
import pickle
from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd
import yaml

from enginora.metrics import get_metric, MetricsConfig
from enginora.selector import get_selector, SelectorConfig
from enginora.utils.mlflow import MlflowManager
from enginora.utils.utils import Stage
from enginora.utils.utils import flatten_dict


@dataclass
class DatasetConfig:
    dataset_path: str

    def load_dataset(self) -> pd.DataFrame:
        # TODO: what columns should be accept as text/target (?)
        # TODO: maybe create ids if not provided (?)
        return pd.read_csv(self.dataset_path, header=None, names=["id", "text", "label"])


@dataclass
class DatasetConfigWithSelectors(DatasetConfig):
    random_state: int = 123
    selectors: List[SelectorConfig] = None

    def load_dataset(self) -> pd.DataFrame:
        df = super().load_dataset()

        # TODO: change names of variables that are single letters
        for t in self.selectors:
            t.args["random_state"] = self.random_state
            selector = get_selector(t.name)(**t.args)
            df = selector.select(df)
        return df


@dataclass
class WithMetrics:
    results_file: str
    metrics_file: str = "./results/metrics.yaml"
    metrics: List[MetricsConfig] = None
    stage: Stage = field(init=False)

    def compute_metrics(self, predictions) -> Dict[str, float]:
        proba_predictions, predictions, true_labels = predictions[0], predictions[0].argmax(1), predictions[1]

        # FIXME: wrong typing here!!!
        return flatten_dict(
            {
                metric.name: get_metric(metric.name)(
                    true_labels, predictions, **metric.args, proba_predictions=proba_predictions, stage=self.stage
                )
                for metric in self.metrics
            }
        )

    def save_results(self, results, mlflow_manager: MlflowManager):
        with open(self.results_file, "wb") as file:
            pickle.dump(results, file)

        self._dump_metrics(results.metrics)
        if mlflow_manager is not None:
            self._log_metrics_to_mlflow(results.metrics, mlflow_manager)

    def load_results(self):
        with open(self.results_file, "rb") as file:
            return pickle.load(file)

    def load_metrics(self) -> Dict:
        with open(self.metrics_file, "r") as file:
            return yaml.safe_load(file)

    def _dump_metrics(self, metrics):
        file_content = {}
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as file:
                file_content = yaml.safe_load(file)

        file_content[self.stage.value] = metrics

        with open(self.metrics_file, "w") as file:
            yaml.dump(file_content, file)

    def _log_metrics_to_mlflow(self, metrics, mlflow_manager: MlflowManager):
        """needs to be invoked strictly after computing and saving metrics - therefore, after saving predictions"""
        metrics_name_with_stage = dict(
            [(metric_name + "_" + self.stage.__str__(), metric_value) for metric_name, metric_value in metrics.items()]
        )
        mlflow_manager.log_metrics(metrics_name_with_stage)
