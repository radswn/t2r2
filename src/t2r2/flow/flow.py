from typing import Dict, List

import mlflow
import torch
from transformers import Trainer
from transformers.utils import logging

from t2r2.config import Config, get_config
from t2r2.dataset import TrainingConfig
from t2r2.flow.dataset import TextDataset
from t2r2.flow.trainer import MetricsOnTrainingSetCallback
from t2r2.utils.curriculum_learning import OrderedTrainer
from t2r2.utils.mlflow import MlflowManager
from t2r2.utils.repo import repo


class T2R2:
    def __init__(self, config_path="./config.yaml"):
        self.config_path = config_path

    # LOOP *************************************************************************************************************

    def loop(self) -> Dict:
        config = get_config(self.config_path)

        logging.set_verbosity_error()

        tokenizer = config.model.create_tokenizer()
        model = config.model.create_model()

        # train, test
        control_results, test_results, train_results = self._train_test_track(config, model, tokenizer)

        # record
        config.dvc.add(self.config_path, config.training, config.testing, config.control, config.model)

        # and repeat!
        return {
            "control_results": control_results,
            "test_results": test_results,
            "train_results": train_results,
        }

    def _train_test_track(self, config, model, tokenizer):
        if config.mlflow:
            mlflow_manager = MlflowManager(config.mlflow)
            experiment_id = mlflow_manager.mlflow_create_experiment()

            with mlflow.start_run(experiment_id=experiment_id) as run:
                train_results, test_results, control_results = self._train_test(
                    model, tokenizer, config, mlflow_manager
                )
        else:
            train_results, test_results, control_results = self._train_test(model, tokenizer, config)
        return control_results, test_results, train_results

    def _train_test(self, model, tokenizer, config: Config, mlflow_manager: MlflowManager = None):
        datasets = self._get_datasets(config, tokenizer, mlflow_manager)
        trainer = self._get_trainer(config.training, datasets, model)

        train_results = self._train(config, mlflow_manager, trainer)
        test_results = self._test(config, datasets, mlflow_manager, trainer)
        control_results = self._control(config, datasets, mlflow_manager, trainer)

        return train_results.metrics, test_results.metrics, control_results.metrics

    def _train(self, config, mlflow_manager, trainer):
        train_results = trainer.train()
        train_results.metrics["history"] = trainer.state.log_history[:-1]  # last logs are already contained in metrics
        config.training.save_results(train_results, mlflow_manager)
        config.model.save_model(trainer)
        return train_results

    def _test(self, config, datasets, mlflow_manager, trainer):
        test_results = trainer.predict(datasets["test"])
        config.testing.save_results(test_results, mlflow_manager)
        return test_results

    def _control(self, config, datasets, mlflow_manager, trainer):
        control_results = trainer.predict(datasets["control"], metric_key_prefix="control")
        config.control.save_results(control_results, mlflow_manager)
        return control_results

    def _get_trainer(self, training_config: TrainingConfig, datasets: Dict[str, TextDataset], model) -> Trainer:
        kwargs = {
            "model": model,
            "train_dataset": datasets["train"],
            "eval_dataset": datasets["validation"],
            "compute_metrics": training_config.compute_metrics,
            "args": training_config.get_training_args(),
            "callbacks": training_config.get_callbacks(),
        }

        if datasets["train"].order is not None:
            return OrderedTrainer(order=datasets["train"].order, **kwargs)

        trainer = Trainer(**kwargs)
        trainer.add_callback(MetricsOnTrainingSetCallback(trainer))
        return trainer

    def _get_datasets(
        self,
        config: Config,
        tokenizer,
        mlflow_manager: MlflowManager,
    ) -> Dict[str, TextDataset]:
        training_dataset, validation_dataset = config.training.load_dataset()
        data = {
            "train": training_dataset[:1000],
            "validation": validation_dataset[:200],
            "test": config.testing.load_dataset()[:200],
            "control": config.control.load_dataset()[:200],
        }

        if mlflow_manager is not None:
            mlflow_manager.log_data(data)

        tokens = {dataset_type: tokenizer(dataset["text"].tolist()) for dataset_type, dataset in data.items()}
        labels = {dataset_type: torch.tensor(dataset["label"].tolist()) for dataset_type, dataset in data.items()}

        return {
            dataset_type: TextDataset(
                tokens[dataset_type],
                labels[dataset_type],
                training_dataset["order"].to_list()
                if dataset_type == "train" and "order" in training_dataset
                else None,
            )
            for dataset_type in data.keys()
        }

    # LOOP END *********************************************************************************************************

    def get_metrics(self) -> List[Dict]:
        config = get_config(self.config_path)

        deduplicated_metrics = {
            config.training.metrics_file: config.training,
            config.testing.metrics_file: config.testing,
            config.control.metrics_file: config.control,
        }

        return [cfg.load_metrics() for cfg in deduplicated_metrics.values()]

    # DVC **************************************************************************************************************
    def init(self):
        """
        If dvc enabled then initialize repository.
        Always store metrics for experiment tracking.
        Storing datasets, results and model is optional.
        """
        config = get_config(self.config_path)

        if config.dvc:
            repo.init(
                self.config_path,
                [config.training.metrics_file, config.testing.metrics_file, config.control.metrics_file],
            )

    def dvc_checkout(self):
        repo.checkout()

    def dvc_metrics(self):
        repo.metrics_diff()

    def dvc_params(self):
        repo.params_diff()

    # DVC END **********************************************************************************************************
