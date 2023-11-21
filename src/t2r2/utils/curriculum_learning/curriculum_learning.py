from typing import Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class OrderedSampler(Sampler):
    def __init__(self, order, data_source: Optional[Sized] = None):
        super().__init__(data_source)
        self.order = order

    def __iter__(self):
        return iter(sorted(range(len(self.order)), key=lambda i: self.order[i]))

    def __len__(self):
        return len(self.order)


class OrderedTrainer(Trainer):
    def __init__(
        self,
        order,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.order = order

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.order is not None:
            return OrderedSampler(self.order)
        else:
            return super()._get_train_sampler()
