from typing import Dict, Tuple

import torch
import yaml
from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, IntervalStrategy, Trainer

from enginora.dataset import ControlConfig, TrainingConfig, TestConfig
from enginora.model import ModelConfig


def loop(config_path='./config.yaml') -> Dict:
    model_config, training_config, test_config, control_config = get_configurations(config_path)

    tokenizer = model_config.create_tokenizer()
    model = model_config.create_model()

    datasets = get_datasets(training_config, control_config, test_config, tokenizer)
    trainer = get_trainer(training_config, datasets, model)

    train_results = trainer.train()

    test_results = trainer.predict(datasets['test'])
    test_config.save_predictions(test_results)

    control_results = trainer.predict(datasets['control'])
    control_config.save_predictions(control_results)

    return {
        'train_results': train_results,
        'test_results': test_results.metrics,
        'control_results': control_results.metrics,
        'test_set_metrics': test_config.compute_metrics(),  # FIXME: is it really needed?
        'control_set_metrics': control_config.compute_metrics(),  # FIXME: is it really needed?
    }


def get_configurations(path: str) -> Tuple[ModelConfig, TrainingConfig, TestConfig, ControlConfig]:
    with open(path, 'r') as stream:
        configuration = yaml.safe_load(stream)

    model_config = ModelConfig(**configuration['model'])
    training_config = TrainingConfig(**configuration['training'])
    test_config = TestConfig(**configuration['testing'])
    control_config = ControlConfig(**configuration['control'])

    return model_config, training_config, test_config, control_config


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
            'input_ids': self.input_ids[i],
            'attention_mask': self.attention_mask[i],
            'token_type_ids': self.token_type_ids[i],
            'labels': self.y[i]
        }


def get_datasets(training_config: TrainingConfig, control_config: ControlConfig, test_config: TestConfig,
                 tokenizer) -> Dict[str, TextDataset]:
    data = {
        'train': training_config.load_dataset()[0],
        'validation': training_config.load_dataset()[1],
        'test': test_config.load_dataset(),
        'control': control_config.load_dataset(),
    }
    tokens = {
        dataset_type: tokenizer(dataset['text'].tolist())
        for dataset_type, dataset in data.items()
    }
    labels = {
        dataset_type: torch.tensor(dataset['label'].tolist())
        for dataset_type, dataset in data.items()
    }

    return {
        dataset_type: TextDataset(tokens[dataset_type], labels[dataset_type])
        for dataset_type in data.keys()
    }


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
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        compute_metrics=training_config.compute_metrics,
        args=get_training_args(training_config)
    )
