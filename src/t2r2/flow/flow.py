from typing import Dict, List

import mlflow
from transformers.utils import logging

from t2r2.config import Config, get_config
from t2r2.flow.dataset import get_datasets
from t2r2.flow.trainer import get_trainer
from t2r2.utils.mlflow import MlflowManager


def get_metrics(config_path="./config.yaml") -> List[Dict]:
    config = get_config(config_path)

    deduplicated_metrics = {
        config.training.metrics_file: config.training,
        config.testing.metrics_file: config.testing,
        config.control.metrics_file: config.control,
    }

    return [cfg.load_metrics() for cfg in deduplicated_metrics.values()]


def loop(config_path="./config.yaml") -> Dict:
    config = get_config(config_path)

    logging.set_verbosity_error()
    tokenizer = config.model.create_tokenizer()
    model = config.model.create_model()

    # train, test
    control_results, test_results, train_results = _train_test_track(config, model, tokenizer)

    # record
    config.dvc.add(config_path, config.training, config.testing, config.control, config.model)

    # and repeat!
    return {
        "train_results": train_results,
        "test_results": test_results,
        "control_results": control_results,
    }


def _train_test_track(config, model, tokenizer):
    if config.mlflow:
        mlflow_manager = MlflowManager(config.mlflow)
        experiment_id = mlflow_manager.mlflow_create_experiment()

        with mlflow.start_run(experiment_id=experiment_id) as run:
            train_results, test_results, control_results = _train_test(model, tokenizer, config, mlflow_manager)
    else:
        train_results, test_results, control_results = _train_test(model, tokenizer, config)
    return control_results, test_results, train_results


def _train_test(model, tokenizer, config: Config, mlflow_manager: MlflowManager = None):
    datasets = get_datasets(config, tokenizer, mlflow_manager)
    trainer = get_trainer(config.training, datasets, model)

    train_results = trainer.train()
    config.training.save_results(train_results, mlflow_manager)

    test_results = trainer.predict(datasets["test"])
    config.testing.save_results(test_results, mlflow_manager)

    control_results = trainer.predict(datasets["control"])
    config.control.save_results(control_results, mlflow_manager)

    return train_results.metrics, test_results.metrics, control_results.metrics
