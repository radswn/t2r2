from copy import deepcopy
from typing import Dict

from transformers import Trainer, TrainerCallback

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

    trainer = Trainer(**kwargs)
    trainer.add_callback(MetricsOnTrainingSetCallback(trainer))
    return trainer


class MetricsOnTrainingSetCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
