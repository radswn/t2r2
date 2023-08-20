from typing import Dict, Tuple

import mlflow
import torch
import yaml
from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, IntervalStrategy, Trainer
from transformers.trainer_utils import PredictionOutput

from enginora.dataset import ControlConfig, TrainingConfig, TestConfig
from enginora.model import ModelConfig
from enginora.utils.mlflow import MlflowManager, MlFlowConfig


def get_train_results(config_path="./config.yaml"):
    _, train_config, _, _, _ = get_configurations(config_path)
    train_results = train_config.load_results()
    return train_results.metrics


def get_test_results(config_path="./config.yaml"):
    _, _, test_config, _, _ = get_configurations(config_path)
    test_results = test_config.load_results()
    return test_results.metrics


def get_control_results(config_path="./config.yaml"):
    _, _, _, control_config, _ = get_configurations(config_path)
    control_results = control_config.load_results()
    return control_results.metrics


def loop(config_path="./config.yaml") -> Dict:
    model_config, training_config, test_config, control_config, mlflow_config = get_configurations(config_path)

    tokenizer = model_config.create_tokenizer()
    model = model_config.create_model()

    if mlflow_config is not None:
        mlflow_manager = MlflowManager(mlflow_config)
        experiment_id = mlflow_manager.mlflow_create_experiment()
        with mlflow.start_run(experiment_id=experiment_id) as run:
            train_results, test_results, control_results = train_and_test(
                model, tokenizer, training_config, test_config, control_config, mlflow_manager)
    else:
        train_results, test_results, control_results = train_and_test(
            model, tokenizer, training_config, test_config, control_config)

    return {
        "train_results": train_results,
        "test_results": test_results.metrics,
        "control_results": control_results.metrics,
    }


def train_and_test(
        model, tokenizer, training_config: TrainingConfig, test_config: TestConfig, control_config: ControlConfig,
        mlflow_manager: MlflowManager = None
) -> Tuple[PredictionOutput, PredictionOutput, PredictionOutput]:
    datasets = get_datasets(training_config, control_config, test_config, tokenizer, mlflow_manager)
    trainer = get_trainer(training_config, datasets, model)

    train_results = trainer.train()
    training_config.save_results(train_results, mlflow_manager)

    test_results = trainer.predict(datasets["test"])
    test_config.save_results(test_results, mlflow_manager)

    control_results = trainer.predict(datasets["control"])
    control_config.save_results(control_results, mlflow_manager)

    return train_results, test_results, control_results


def get_configurations(
        path: str,
) -> Tuple[ModelConfig, TrainingConfig, TestConfig, ControlConfig, MlFlowConfig]:
    with open(path, "r") as stream:
        configuration = yaml.safe_load(stream)

    metrics = configuration["metrics"]
    for config in ["training", "testing", "control"]:
        configuration[config]["metrics"] = metrics

    model_config = ModelConfig(**configuration["model"])
    training_config = TrainingConfig(**configuration["training"])
    test_config = TestConfig(**configuration["testing"])
    control_config = ControlConfig(**configuration["control"])

    mlflow_config = None
    if "mlflow" in configuration:
        mlflow_config = MlFlowConfig(**configuration["mlflow"])

    return model_config, training_config, test_config, control_config, mlflow_config


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


def get_datasets(
        training_config: TrainingConfig, control_config: ControlConfig, test_config: TestConfig, tokenizer,
        mlflow_manager: MlflowManager,
) -> Dict[str, TextDataset]:
    training_dataset, validation_dataset = training_config.load_dataset()
    data = {
        "train": training_dataset,
        "validation": validation_dataset,
        "test": test_config.load_dataset(),
        "control": control_config.load_dataset(),
    }

    if mlflow_manager is not None:
        mlflow_manager.log_data(data)

    tokens = {dataset_type: tokenizer(dataset["text"].tolist()) for dataset_type, dataset in data.items()}
    labels = {dataset_type: torch.tensor(dataset["label"].tolist()) for dataset_type, dataset in data.items()}

    return {dataset_type: TextDataset(tokens[dataset_type], labels[dataset_type]) for dataset_type in data.keys()}


def get_training_args(training_config: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=training_config.output_dir,
        learning_rate=training_config.learning_rate,
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        load_best_model_at_end=True,
        metric_for_best_model=training_config.metric_for_best_model,
        num_train_epochs=training_config.epochs,
    )


def get_trainer(training_config: TrainingConfig, datasets: Dict[str, TextDataset], model) -> Trainer:
    return Trainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=training_config.compute_metrics,
        args=get_training_args(training_config),
    )
