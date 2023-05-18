from dataclasses import dataclass

from sklearn.metrics import accuracy_score


def get_metric(name: str):
    # TODO: add more metrics
    metrics = {
        'accuracy': accuracy_score,
    }

    return metrics[name]


@dataclass
class MetricsConfig:
    name: str
