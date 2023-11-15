from typing import Dict, List, Tuple

import mlflow
import torch
import yaml
from torch.utils.data.dataset import Dataset
from transformers import IntervalStrategy, Trainer, TrainingArguments

from t2r2.dataset import ControlConfig, TestConfig, TrainingConfig
from t2r2.model import ModelConfig
from t2r2.utils import repo
from t2r2.utils.mlflow import MlFlowConfig, MlflowManager
from t2r2.utils.repo import DvcConfig


def init(config_path="./config.yaml"):
    """
    If dvc enabled then initialize repository.
    Always store metrics for experiment tracking.
    Storing datasets, results and model is optional.
    """
    _, training_config, test_config, control_config, _, dvc_config = get_configurations(config_path)
    if dvc_config.enabled:
        repo.init(config_path, [training_config.metrics_file, test_config.metrics_file, control_config.metrics_file])


def dvc_checkout(config_path="./config.yaml"):
    repo.checkout()


def dvc_metrics(config_path="./config.yaml"):
    repo.metrics_diff()


def dvc_params(config_path="./config.yaml"):
    repo.params_diff()


def get_metrics(config_path="./config.yaml") -> List[Dict]:
    _, train_config, test_config, control_config, _, _ = get_configurations(config_path)
    deduplicated_metrics = {
        # FIXME is it supposed to be "train_config" everywhere?
        train_config.metrics_file: train_config,
        test_config.metrics_file: train_config,
        control_config.metrics_file: train_config,
    }
    return [cfg.load_metrics() for cfg in deduplicated_metrics]


def loop(config_path="./config.yaml") -> Dict:
    model_config, training_config, test_config, control_config, mlflow_config, dvc_config = get_configurations(
        config_path
    )

    tokenizer = model_config.create_tokenizer()
    model = model_config.create_model()

    if mlflow_config is not None:
        mlflow_manager = MlflowManager(mlflow_config)
        experiment_id = mlflow_manager.mlflow_create_experiment()
        with mlflow.start_run(experiment_id=experiment_id) as run:
            train_results, test_results, control_results = train_and_test(
                model, tokenizer, training_config, test_config, control_config, mlflow_manager
            )
    else:
        train_results, test_results, control_results = train_and_test(
            model, tokenizer, training_config, test_config, control_config
        )

    dvc_config.add(config_path, training_config, test_config, control_config, model_config)

    return {
        "train_results": train_results,
        "test_results": test_results,
        "control_results": control_results,
    }


def train_and_test(
    model,
    tokenizer,
    training_config: TrainingConfig,
    test_config: TestConfig,
    control_config: ControlConfig,
    mlflow_manager: MlflowManager = None,
):
    datasets = get_datasets(training_config, control_config, test_config, tokenizer, mlflow_manager)
    trainer = get_trainer(training_config, datasets, model)

    train_results = trainer.train()
    training_config.save_results(train_results, mlflow_manager)

    test_results = trainer.predict(datasets["test"])
    test_config.save_results(test_results, mlflow_manager)

    control_results = trainer.predict(datasets["control"])
    control_config.save_results(control_results, mlflow_manager)

    return train_results.metrics, test_results.metrics, control_results.metrics


def get_configurations(
    path: str,
) -> Tuple[ModelConfig, TrainingConfig, TestConfig, ControlConfig, MlFlowConfig, DvcConfig]:
    with open(path, "r") as stream:
        configuration = yaml.safe_load(stream)

    random_state = configuration.get("random_state", 123)
    configuration["training"]["random_state"] = random_state
    configuration["testing"]["random_state"] = random_state
    
    device = configuration.get("device", "cuda:0" if torch.cuda.is_available() else None)
   
    set_seed_and_device(random_state, device)

    metrics = configuration["metrics"]
    for config in ["training", "testing", "control"]:
        configuration[config]["metrics"] = metrics

    model_config = ModelConfig(**configuration["model"])
    training_config = TrainingConfig(**configuration["training"])
    test_config = TestConfig(**configuration["testing"])
    control_config = ControlConfig(**configuration["control"])

    mlflow_config = None
    if "mlflow" in configuration:
        configuration["mlflow"]["random_state"] = random_state
        mlflow_config = MlFlowConfig(**configuration["mlflow"])
    dvc_config = DvcConfig() if "dvc" not in configuration else DvcConfig(**configuration["dvc"])

    return model_config, training_config, test_config, control_config, mlflow_config, dvc_config


def set_seed_and_device(random_state: int, device: str):
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    if device and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(device)
        print("[T2R2] Torch device set to CUDA.")
    elif device and not torch.cuda.is_available():
        print("[T2R2] CUDA not available.")


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
    training_config: TrainingConfig,
    control_config: ControlConfig,
    test_config: TestConfig,
    tokenizer,
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


def get_trainer(training_config: TrainingConfig, datasets: Dict[str, TextDataset], model) -> Trainer:
    return Trainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=training_config.compute_metrics,
        args=training_config.get_training_args(),
        callbacks=training_config.get_callbacks(),
    )
