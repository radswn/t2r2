from t2r2.config import get_config
from t2r2.utils.repo import repo


def init(config_path="./config.yaml"):
    """
    If dvc enabled then initialize repository.
    Always store metrics for experiment tracking.
    Storing datasets, results and model is optional.
    """
    config = get_config(config_path)

    if config.dvc:
        repo.init(config_path, [config.training.metrics_file, config.testing.metrics_file, config.control.metrics_file])


def dvc_checkout():
    repo.checkout()


def dvc_metrics():
    repo.metrics_diff()


def dvc_params():
    repo.params_diff()
