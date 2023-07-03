from dataclasses import dataclass
from typing import Dict, Any
from enginora.metrics.slicing_scoring import slicing_scores
from sklearn.metrics import accuracy_score, f1_score

def get_metric(name: str):
    metrics = {"accuracy": accuracy_score, "f1": f1_score, "slicing_scores": slicing_scores}
    return metrics[name]

@dataclass
class MetricsConfig:
    name: str
    args: Dict[str, Any]

    def __post_init__(self):
        if self.args is None:
            self.args = dict()