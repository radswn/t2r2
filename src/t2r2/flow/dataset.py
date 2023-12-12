import torch
from torch.utils.data import Dataset


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
