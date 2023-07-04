from dataclasses import dataclass
from typing import Dict, Any
from enginora.metrics.slicing_scoring import slicing_scores
from sklearn.metrics import accuracy_score, f1_score
from enginora.utils import ignore_unmatched_kwargs


def get_metric(name: str):
    metrics = {
        "accuracy": ignore_unmatched_kwargs(accuracy_score),
        "f1": ignore_unmatched_kwargs(f1_score),
        "slicing": ignore_unmatched_kwargs(slicing_scores),
    }
    return metrics[name]


@dataclass
class MetricsConfig:
    name: str
    args: Dict[str, Any]

    def __post_init__(self):
        if self.args is None:
            self.args = dict()
