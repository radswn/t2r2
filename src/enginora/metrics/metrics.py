from dataclasses import dataclass
from typing import Dict, Any

from sklearn.metrics import accuracy_score, f1_score


def get_metric(name: str):
    metrics = {
        'accuracy': accuracy_score,
        'f1': f1_score
    }

    return metrics[name]


@dataclass
class MetricsConfig:
    name: str
    args: Dict[str, Any]

    def __post_init__(self):
        if self.args is None:
            self.args = dict()
