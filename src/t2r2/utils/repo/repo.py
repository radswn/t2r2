import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, List

import yaml

from t2r2.dataset import ControlConfig, TestConfig, TrainingConfig
from t2r2.model import ModelConfig


@dataclass
class _DvcDatasetConfig:
    track_dataset: bool = True
    track_results: bool = True


@dataclass
class DvcConfig:
    enabled: bool = False
    training: _DvcDatasetConfig = _DvcDatasetConfig()
    test: _DvcDatasetConfig = _DvcDatasetConfig()
    control: _DvcDatasetConfig = _DvcDatasetConfig()
    track_model: bool = True

    def add(
        self,
        config_path: str,
        training_config: TrainingConfig,
        test_config: TestConfig,
        control_config: ControlConfig,
        model_config: ModelConfig,
    ):
        if self.enabled:
            _add(config_path, training_config.metrics_file, test_config.metrics_file, control_config.metrics_file)

            # TODO: model tracking (?)

            if self.training.track_dataset:
                _add(training_config.dataset_path)
            if self.training.track_results:
                _add(training_config.results_file)

            if self.test.track_dataset:
                _add(test_config.dataset_path)
            if self.test.track_results:
                _add(test_config.results_file)

            if self.control.track_dataset:
                _add(control_config.dataset_path)
            if self.control.track_results:
                _add(control_config.results_file)


LOGGER = logging.getLogger("DVC repo")


def init(config_file: str, metric_files: List[str]):
    with open("dvc.yaml", "w") as file:
        yaml.dump(_get_dvc_yaml_content(config_file, metric_files), file)

    _run("dvc init")
    _run("dvc config core.autostage true")


def checkout():
    _run("dvc checkout")


def metrics_diff():
    _run("dvc metrics diff")


def params_diff():
    _run("dvc params diff")


def _add(*paths: str):
    _run(f"dvc add {' '.join(paths)}")


def _run(cmd: str):
    LOGGER.info(f"Running: {cmd}")
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE)
    if result.stderr is not None:
        LOGGER.error("\n" + result.stderr.decode("utf-8"))
    else:
        LOGGER.info("\n" + result.stdout.decode("utf-8"))


def _get_dvc_yaml_content(config_file: str, metric_files: List[str]) -> Dict:
    return {
        "metrics": list(set(metric_files)),
        "params": [config_file],
    }
