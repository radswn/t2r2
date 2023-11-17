import pytest

from t2r2.flow import check_metric
from t2r2.metrics import MetricsConfig

def test_valid_metric():
    metric_dict = {"name": "accuracy_score", "args": {"normalize": True}}
    metric = MetricsConfig(**metric_dict)

    check_metric(metric)


def test_invalid_metric_name():
    metric_dict = {"name": "blablabla_score"}
    metric = MetricsConfig(**metric_dict)

    with pytest.raises(ValueError, match="does not exist"):
        check_metric(metric)


def test_not_handled_metric_name():
    metric_dict = {"name": "dcg_score"}
    metric = MetricsConfig(**metric_dict)

    with pytest.raises(ValueError, match="not handled by T2R2"):
        check_metric(metric)
