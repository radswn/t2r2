import os.path
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Union

import pandas as pd
import yaml

from t2r2.metrics import get_metric, MetricsConfig
from t2r2.selector import get_selector, get_custom_selector, SelectorConfig
from t2r2.utils.mlflow import MlflowManager
from t2r2.utils.utils import Stage
from t2r2.utils.utils import flatten_dict
from t2r2.utils.utils import check_if_directory_exists


@dataclass
class DatasetConfig:
    dataset_path: str
    text_column_id: int
    label_column_id: int
    has_header: bool

    def load_dataset(self) -> pd.DataFrame:
        header = 0 if self.has_header else None

        df = pd.read_csv(self.dataset_path, header=header)
        df = df.iloc[:, [self.text_column_id, self.label_column_id]]
        df.columns = ["text", "label"]

        return df


@dataclass
class DatasetConfigWithSelectors(DatasetConfig):
    random_state: int = 123
    selectors: List[SelectorConfig] = None

    def load_dataset(self) -> pd.DataFrame:
        df = super().load_dataset()

        for selector_config in self.selectors:
            selector_config.args["random_state"] = self.random_state
            if "module_path" in selector_config.args:
                selector = get_custom_selector(selector_config.args["module_path"], selector_config.name)(
                    **selector_config.args
                )
            else:
                selector = get_selector(selector_config.name)(**selector_config.args)
            df = selector.select(df)

        return df


@dataclass
class WithMetrics:
    results_file: str
    metrics_file: str = "metrics.yaml"
    output_dir: str = "./results"
    metrics: List[MetricsConfig] = None
    stage: Stage = field(init=False)

    def compute_metrics(self, outputs) -> MutableMapping:
        proba_predictions, predictions, true_labels = outputs[0], outputs[0].argmax(1), outputs[1]

        return flatten_dict(
            {
                metric.name: get_metric(metric.name)(
                    true_labels, predictions, **metric.args, proba_predictions=proba_predictions, stage=self.stage
                )
                for metric in self.metrics
            }
        )

    def save_results(self, results, mlflow_manager: MlflowManager):
        results_path = os.path.join(self.output_dir, self.results_file)
        check_if_directory_exists(results_path)

        with open(results_path, "wb") as file:
            pickle.dump(results_path, file)

        self._dump_metrics(results.metrics)
        if mlflow_manager is not None:
            self._log_metrics_to_mlflow(results.metrics, mlflow_manager)

    def load_results(self):
        results_path = os.path.join(self.output_dir, self.results_file)
        with open(results_path, "rb") as file:
            return pickle.load(file)

    def load_metrics(self) -> Dict:
        metrics_path = os.path.join(self.output_dir, self.metrics_file)
        with open(metrics_path, "r") as file:
            return yaml.safe_load(file)

    def _dump_metrics(self, metrics):
        file_content = {}
        metrics_path = os.path.join(self.output_dir, self.metrics_file)
        check_if_directory_exists(metrics_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as file:
                file_content = yaml.safe_load(file)

        file_content[self.stage.value] = metrics

        with open(metrics_path, "w") as file:
            yaml.dump(file_content, file)

    def _log_metrics_to_mlflow(self, metrics, mlflow_manager: MlflowManager):
        """needs to be invoked strictly after computing and saving metrics - therefore, after saving predictions"""
        metrics_name_with_stage = dict(
            [(metric_name + "_" + self.stage.__str__(), metric_value) for metric_name, metric_value in metrics.items()]
        )
        mlflow_manager.log_metrics(metrics_name_with_stage)
