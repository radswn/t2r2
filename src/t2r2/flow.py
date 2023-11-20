from typing import Dict, List

import mlflow
import torch

from torch.utils.data.dataset import Dataset
from transformers import Trainer

from t2r2.dataset import TrainingConfig
from t2r2.utils import repo
from t2r2.utils.mlflow import MlflowManager
from t2r2.config import Config, load_config


def init(config_path="./config.yaml"):
    """
    If dvc enabled then initialize repository.
    Always store metrics for experiment tracking.
    Storing datasets, results and model is optional.
    """
    config = get_config(config_path)

    if config.dvc:
        repo.init(config_path, [config.training.metrics_file, config.testing.metrics_file, config.control.metrics_file])


def dvc_checkout(config_path="./config.yaml"):
    repo.checkout()


def dvc_metrics(config_path="./config.yaml"):
    repo.metrics_diff()


def dvc_params(config_path="./config.yaml"):
    repo.params_diff()


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

    tokenizer = config.model.create_tokenizer()
    model = config.model.create_model()

    if config.mlflow:
        mlflow_manager = MlflowManager(config.mlflow)
        experiment_id = mlflow_manager.mlflow_create_experiment()

        with mlflow.start_run(experiment_id=experiment_id) as run:
            train_results, test_results, control_results = train_and_test(model, tokenizer, config, mlflow_manager)
    else:
        train_results, test_results, control_results = train_and_test(model, tokenizer, config)

    config.dvc.add(config_path, config.training, config.testing, config.control, config.model)

    return {
        "train_results": train_results,
        "test_results": test_results,
        "control_results": control_results,
    }


def train_and_test(model, tokenizer, config: Config, mlflow_manager: MlflowManager = None):
    datasets = get_datasets(config, tokenizer, mlflow_manager)
    trainer = get_trainer(config.training, datasets, model)

    train_results = trainer.train()
    config.training.save_results(train_results, mlflow_manager)

    test_results = trainer.predict(datasets["test"])
    config.testing.save_results(test_results, mlflow_manager)

    control_results = trainer.predict(datasets["control"])
    config.control.save_results(control_results, mlflow_manager)

    return train_results.metrics, test_results.metrics, control_results.metrics


def set_seed(random_state: int):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True


class TextDataset(Dataset):
    def __init__(self, tokens, labels: torch.Tensor):
        self.input_ids = tokens.input_ids
        self.attention_mask = tokens.attention_mask
        self.token_type_ids = tokens.token_type_ids
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "token_type_ids": self.token_type_ids[i],
            "labels": self.y[i],
        }


def get_config(path: str) -> Config:
    config_dict = load_config(path)
    config = Config(**config_dict)

    if config.random_state is not None:
        set_seed(config.random_state)

    return config


def get_datasets(
    config: Config,
    tokenizer,
    mlflow_manager: MlflowManager,
) -> Dict[str, TextDataset]:
    training_dataset, validation_dataset = config.training.load_dataset()
    data = {
        "train": training_dataset,
        "validation": validation_dataset,
        "test": config.testing.load_dataset(),
        "control": config.control.load_dataset(),
    }

    if mlflow_manager is not None:
        mlflow_manager.log_data(data)

    tokens = {dataset_type: tokenizer(dataset["text"].tolist()) for dataset_type, dataset in data.items()}
    labels = {dataset_type: torch.tensor(dataset["label"].tolist()) for dataset_type, dataset in data.items()}

    return {dataset_type: TextDataset(tokens[dataset_type], labels[dataset_type]) for dataset_type in data.keys()}


def get_trainer(training_config: TrainingConfig, datasets: Dict[str, TextDataset], model) -> Trainer:
    return Trainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=training_config.compute_metrics,
        args=training_config.get_training_args(),
        callbacks=training_config.get_callbacks(),
    )
