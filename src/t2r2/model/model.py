from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ModelConfig:
    model_name: str
    output_dir: str
    num_labels: int = 2
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    output_path: str = "best_model"

    def __post_init__(self):
        self.num_labels = int(self.num_labels)
        self.max_length = int(self.max_length)
        self.truncation = bool(self.truncation)

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return lambda input_tokens: tokenizer(
            input_tokens,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_token_type_ids=True,
        )

    def create_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

    def get_output_path(self):
        return Path(self.output_dir) / self.output_path

    def save_model(self, trainer):
        trainer.save_model(self.get_output_path())
