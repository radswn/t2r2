from dataclasses import dataclass
from typing import Dict, Any

from enginora.metrics.slicing_scoring import slicing_scores
from enginora.utils import ignore_unmatched_kwargs
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    class_likelihood_ratios,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
    brier_score_loss,
    multilabel_confusion_matrix,
)


def get_metric(name: str):
    metrics = {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "class_likelihood_ratios": class_likelihood_ratios,
        "classification_report": classification_report,
        "cohen_kappa_score": cohen_kappa_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "fbeta_score": fbeta_score,
        "hamming_loss": hamming_loss,
        "hinge_loss": hinge_loss,
        "jaccard_score": jaccard_score,
        "log_loss": log_loss,
        "matthews_corrcoef": matthews_corrcoef,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "zero_one_loss": zero_one_loss,
        "brier_score_loss": brier_score_loss,
        "multilabel_confusion_matrix": multilabel_confusion_matrix,
    }
    return ignore_unmatched_kwargs(metrics[name])


@dataclass
class MetricsConfig:
    name: str
    args: Dict[str, Any]

    def __post_init__(self):
        if self.args is None:
            self.args = dict()
