from typing import Dict

import torch
from torch.utils.data import Dataset

from t2r2.config import Config
from t2r2.utils.mlflow import MlflowManager


class TextDataset(Dataset):
    def __init__(self, tokens, labels: torch.Tensor, order=None):
        self.input_ids = tokens.input_ids
        self.attention_mask = tokens.attention_mask
        self.token_type_ids = tokens.token_type_ids
        self.y = labels
        self.order = order

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

    return {
        dataset_type: TextDataset(
            tokens[dataset_type],
            labels[dataset_type],
            training_dataset["order"].to_list() if dataset_type == "train" and "order" in training_dataset else None,
        )
        for dataset_type in data.keys()
    }
