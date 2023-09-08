from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class ModelConfig:
    model_name: str
    num_labels: int
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"

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
        )

    def create_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
