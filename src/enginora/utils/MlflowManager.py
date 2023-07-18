import logging
import mlflow
import torch


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class MlflowManager(metaclass=Singleton):
    def __init__(self, mlflow_config : dict):
        self.experiment_name = mlflow_config['experiment_name']
        self.tags = mlflow_config['tags']
        self.tracking_uri = mlflow_config['tracking_uri']
        self.logger = logging.getLogger(__name__)
        self.set_tracking_uri()

    def mlflow_create_experiment(self) -> str:
        # Create an experiment name, which must be unique and case sensitive
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

    def log_model(self, model, model_name = "model"):
        mlflow.pytorch.log_model(model, model_name)
        self.logger.info("mlflow: logging model")

    def set_tracking_uri(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        self.logger.info("mlflow: Tracking uri set")

    def log_datasets(self, datasets: dict):
        for dataset_context, dataset in datasets.items():
            mlflow.log_input(dataset, context=dataset_context)
        self.logger.info("mlflow: logged datasets")
