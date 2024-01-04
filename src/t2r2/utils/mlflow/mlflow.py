import logging
import os
from dataclasses import dataclass
from typing import Dict

import mlflow
import pandas as pd


@dataclass
class MlFlowConfig:
    experiment_name: str
    tags: dict
    tracking_uri: str
    random_state: int


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MlflowManager(metaclass=Singleton):
    def __init__(self, mlflow_config: MlFlowConfig):
        self.experiment_name = mlflow_config.experiment_name
        self.tags = mlflow_config.tags
        self.tracking_uri = mlflow_config.tracking_uri
        self.logger = logging.getLogger(__name__)
        self.random_state = mlflow_config.random_state
        self.set_tracking_uri()

    @staticmethod
    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MlflowManager.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(MlflowManager.flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    def mlflow_create_experiment(self) -> str:
        experiment = mlflow.set_experiment(self.experiment_name)
        mlflow.set_experiment_tags(self.tags)

        self.logger.info("***SETTING EXPERIMENT***")
        self.logger.info("Name: {}".format(experiment.name))
        self.logger.info("Experiment_id: {}".format(experiment.experiment_id))
        self.logger.info("Random State: {}".format(self.random_state))
        self.logger.info("Artifact Location: {}".format(experiment.artifact_location))
        self.logger.info("Tags: {}".format(experiment.tags))
        self.logger.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
        self.logger.info("Creation timestamp: {}".format(experiment.creation_time))
        return experiment.experiment_id

    def log_metrics(self, metrics: dict):
        self.logger.info("mlflow: flattening :" + " ".join(list(metrics.keys())))
        flattened_metrics = MlflowManager.flatten_dict(metrics)
        self.logger.info("mlflow: logging metrics :" + " ".join(list(flattened_metrics.keys())))
        mlflow.log_metrics(flattened_metrics)

    def log_data(self, data: Dict[str, pd.DataFrame]):
        for key, value in data.items():
            mlflow.log_input(mlflow.data.from_pandas(value), context=key)
        self.logger.info("mlflow: Dataset logged")
        self.log_dataset_synopsis(data)

    def log_model(self, model, model_name="model"):
        mlflow.pytorch.log_model(model, model_name)
        self.logger.info("mlflow: logging model")

    def set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.logger.info("mlflow: Tracking uri set")

    def log_dataset_synopsis(self, data: Dict[str, pd.DataFrame]):
        """working for csv only right now"""
        for dataset_name, dataset_df in data.items():
            dataset_df.describe().to_html(f"dataset_{dataset_name}.html")
            mlflow.log_artifact(f"dataset_{dataset_name}.html", f"stat_descriptive_{dataset_name}")
            os.remove(f"dataset_{dataset_name}.html")
        self.logger.info("mlflow: Dataset synopsis logged")
