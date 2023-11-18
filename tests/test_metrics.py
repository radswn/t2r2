import pytest

from t2r2.metrics import MetricsConfig


def test_valid_metric():
    metric_dict = {"name": "accuracy_score", "args": {"normalize": True}}
    _ = MetricsConfig(**metric_dict)


def test_invalid_metric_name():
    metric_dict = {"name": "blablabla_score"}

    with pytest.raises(ValueError, match="does not exist"):
        _ = MetricsConfig(**metric_dict)


def test_not_handled_metric_name():
    metric_dict = {"name": "dcg_score"}

    with pytest.raises(ValueError, match="not handled by T2R2"):
        _ = MetricsConfig(**metric_dict)
