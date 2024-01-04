import os.path
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping

import pandas as pd
import yaml

from t2r2.metrics import MetricsConfig, get_metric
from t2r2.selector import SelectorConfig, get_custom_selector, get_selector
from t2r2.utils import Stage, check_if_directory_exists, flatten_dict
from t2r2.utils.mlflow import MlflowManager
from typing import Optional


@dataclass
class DatasetConfig:
    dataset_path: str
    text_column_id: Optional[int] = None
    label_column_id: Optional[int] = None
    has_header: Optional[bool] = None

    def load_dataset(self) -> pd.DataFrame:
        header = 0 if self.has_header else None

        df = pd.read_csv(self.dataset_path, header=header)
        df = df.iloc[:, [self.text_column_id, self.label_column_id]]
        df.columns = ["text", "label"]

        return df


@dataclass
class DatasetConfigWithSelectors(DatasetConfig):
    random_state: int = None
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
    output_dir: str
    results_file: str
    metrics_file: str = "metrics.yaml"
    metrics: List[MetricsConfig] = None
    stage: Stage = field(init=False)

    def __post_init__(self):
        self._propagate_output_path()

    def compute_metrics(self, outputs) -> MutableMapping:
        proba_predictions, predictions, true_labels = outputs[0], outputs[0].argmax(1), outputs[1]

        self._modify_args_of_metrics(proba_predictions)

        return flatten_dict(
            {
                metric.name: get_metric(metric.name)(
                    true_labels,
                    predictions,
                    **metric.args,
                    # FIXME: this doesn't work without ignore_unmatched_kwargs
                    # proba_predictions=proba_predictions, stage=self.stage
                )
                for metric in self.metrics
            }
        )

    def save_results(self, results, mlflow_manager: MlflowManager):
        check_if_directory_exists(self.results_file)

        with open(self.results_file, "wb") as file:
            pickle.dump(self.results_file, file)

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
        check_if_directory_exists(self.metrics_file)

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

    def _modify_args_of_metrics(self, proba_predictions):
        for metric in self.metrics:
            if metric.name == "slicing_scores":
                metric.args.update({"proba_predictions": proba_predictions, "stage": self.stage})

    def _propagate_output_path(self):
        self.results_file = os.path.join(self.output_dir, self.results_file)
        self.metrics_file = os.path.join(self.output_dir, self.metrics_file)
