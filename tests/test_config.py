import pytest
from pathlib import Path

from t2r2.config import load_config, Config


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


def test_column_propagation(test_config_dict):
    test_config_dict["data"]["text_column_id"] = 1
    test_config_dict["data"]["label_column_id"] = 2

    test_config_dict["testing"]["text_column_id"] = 0
    test_config_dict["testing"]["label_column_id"] = 1

    cfg = Config(**test_config_dict)

    assert cfg.training.text_column_id == 1
    assert cfg.training.label_column_id == 2

    assert cfg.testing.text_column_id == 0
    assert cfg.testing.label_column_id == 1

    assert cfg.control.text_column_id == 1
    assert cfg.control.label_column_id == 2


def test_random_state_propagation(test_config_dict):
    random_state = 123
    test_config_dict["random_state"] = random_state
    cfg = Config(**test_config_dict)

    assert cfg.testing.random_state == cfg.training.random_state == random_state


def test_output_dir_propagation(test_config_dict):
    output_dir = "123"
    test_config_dict["data"]["output_dir"] = output_dir
    cfg = Config(**test_config_dict)

    assert cfg.testing.output_dir == cfg.training.output_dir == output_dir


def test_metrics_propagation(test_config_dict):
    metrics = [{"name": "f1_score", "args": {"average": "macro"}}, {"name": "accuracy_score"}]
    test_config_dict["metrics"] = metrics
    cfg = Config(**test_config_dict)

    assert cfg.training.metrics == cfg.control.metrics == cfg.testing.metrics
    assert cfg.training.metrics[1].name == "accuracy_score"


def test_unexpected_argument(test_config_dict):
    test_config_dict["some_random_argument"] = "that should not be here"

    with pytest.raises(TypeError):
        cfg = Config(**test_config_dict)


def test_wrong_argument_type(test_config_dict):
    test_config_dict["model"]["max_length"] = "definitely_not_an_int"

    with pytest.raises(ValueError, match="invalid literal"):
        cfg = Config(**test_config_dict)


def test_invalid_best_model_metric(test_config_dict):
    test_config_dict["training"]["metric_for_best_model"] = "accuracy_score"

    with pytest.raises(ValueError, match="not defined in metrics"):
        cfg = Config(**test_config_dict)
