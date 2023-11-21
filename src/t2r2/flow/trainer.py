from typing import Dict

from transformers import Trainer

from t2r2.dataset import TrainingConfig
from t2r2.flow.dataset import TextDataset
from t2r2.utils.curriculum_learning import OrderedTrainer


def get_trainer(training_config: TrainingConfig, datasets: Dict[str, TextDataset], model) -> Trainer:
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

    return Trainer(**kwargs)
