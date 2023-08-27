import logging
import mlflow
import pandas as pd
import os
from enginora.utils.mlflow.MlFlowConfig import MlFlowConfig
from mlflow.models.signature import Schema

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
        self.registered_model_name = mlflow_config.registered_model_name
        self.set_tracking_uri()

    def mlflow_create_experiment(self) -> str:
        experiment = mlflow.set_experiment(self.experiment_name)
        mlflow.set_experiment_tags(self.tags)

        self.logger.info("***SETTING EXPERIMENT***")
        self.logger.info("Name: {}".format(experiment.name))
        self.logger.info("Experiment_id: {}".format(experiment.experiment_id))
        self.logger.info("Artifact Location: {}".format(experiment.artifact_location))
        self.logger.info("Tags: {}".format(experiment.tags))
        self.logger.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
        self.logger.info("Creation timestamp: {}".format(experiment.creation_time))
        return experiment.experiment_id

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        self.logger.info("mlflow: logging metrics :" + " ".join(list(metrics.keys())))

    def log_data(self, data: dict[str, pd.DataFrame]):
        for key, value in data.items():
            mlflow.log_input(mlflow.data.from_pandas(value), context=key)
        self.logger.info("mlflow: Dataset logged")
        self.log_dataset_synopsis(data)

    def log_model(self, model, input_schema, model_name="model"):
        signature = mlflow.models.ModelSignature(inputs = input_schema)
        mlflow.pytorch.log_model(model, 
                                    model_name, 
                                    registered_model_name = self.registered_model_name,
                                    signature=signature)
        self.logger.info("mlflow: logging model")
        

    def set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.logger.info("mlflow: Tracking uri set")

    def log_dataset_synopsis(self, data: dict[str, pd.DataFrame]):
        """working for csv only right now"""
        for dataset_name, dataset_df in data.items():
            dataset_df.describe().to_html(f"dataset_{dataset_name}.html")
            mlflow.log_artifact(f"dataset_{dataset_name}.html", f"stat_descriptive_{dataset_name}")
            os.remove(f"dataset_{dataset_name}.html")
        self.logger.info("mlflow: Dataset synopsis logged")
