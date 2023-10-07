import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SavePredictionsCallback(TrainerCallback):
    def __init__(self):
        self.predictions = {}
        self.inputs = {}
        self.labels = {}
        self.epoch = 0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.epoch += 1

    def on_prediction_step(self, args, state, control, **kwargs):
        kwargs["model"].eval()

        predictions = []
        labels = []
        with torch.no_grad():
            for i in range(len(kwargs["train_dataloader"].dataset.input_ids)):
                input_id = kwargs["train_dataloader"].dataset.input_ids[i]
                attention_mask = kwargs["train_dataloader"].dataset.attention_mask[i]
                token_type_id = kwargs["train_dataloader"].dataset.token_type_ids[i]
                label = kwargs["train_dataloader"].dataset.y[i]

                input_id = input_id.unsqueeze(0).clone().detach()  # Add batch dimension
                attention_mask = attention_mask.unsqueeze(0).clone().detach()  # Add batch dimension
                token_type_id = token_type_id.unsqueeze(0).clone().detach()  # Add batch dimension

                outputs = kwargs["model"](input_id, attention_mask=attention_mask, token_type_ids=token_type_id)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions.append(probabilities)
                labels.append(int(label))
            self.predictions[self.epoch] = predictions
            self.labels[self.epoch] = labels

    def get_predictions(self):
        return self.predictions

    def get_inputs(self):
        return self.inputs

    def get_labels(self):
        assert all([self.labels[1] == self.labels[epoch] for epoch in range(1, self.epoch)])
        return list(self.labels[1])

    def get_epochs(self):
        return self.epoch
