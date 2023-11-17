from t2r2.flow import load_config, Config
from pathlib import Path
import pytest


@pytest.fixture
def test_config_dict():
    test_config_path = Path(__file__).parent.resolve() / "config.yaml"
    config_dict = load_config(test_config_path)

    return config_dict


def test_valid_config(test_config_dict):
    cfg = Config(**test_config_dict)

    assert cfg.random_state == None
    assert cfg.mlflow == None
    assert cfg.dvc != None and cfg.dvc.enabled == False


def test_random_state_propagation(test_config_dict):
    random_state = 123
    test_config_dict["random_state"] = random_state
    cfg = Config(**test_config_dict)

    assert cfg.testing.random_state == cfg.training.random_state == random_state


def test_metrics_propagation(test_config_dict):
    metrics = [{"name": "f1_score", "args": {"average": "macro"}}, {"name": "accuracy_score"}]
    test_config_dict["metrics"] = metrics
    cfg = Config(**test_config_dict)

    assert cfg.training.metrics == cfg.control.metrics == cfg.testing.metrics
    assert cfg.training.metrics[1].name == "accuracy_score"
