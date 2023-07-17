import inspect
import functools
from typing import Union  # in later version of python simly replace with |
import os
import enum
import mlflow
from pathlib import Path


def ignore_unmatched_kwargs(f):
    """
    Make function ignore unmatched kwargs.

    If the function already has the catch all **kwargs, do nothing.
    """
    if contains_var_kwarg(f):
        return f

    @functools.wraps(f)
    def inner(*args, **kwargs):
        filtered_kwargs = {key: value for key, value in kwargs.items() if is_kwarg_of(key, f)}
        return f(*args, **filtered_kwargs)

    return inner


def contains_var_kwarg(f):
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(f).parameters.values())


def is_kwarg_of(key, f):
    param = inspect.signature(f).parameters.get(key, False)
    return param and (
        param.kind is inspect.Parameter.KEYWORD_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


def construct_results_file(
    slice_file: Union[str, None],
    stage: Union[str, None],
    base_directory: Union[str, None],
    default_file_name: Union[str, None],
) -> str:
    """Constructs filename where slices are stored"""
    if slice_file is None:
        slice_file = os.path.join(base_directory, str(stage.value) + "_" + default_file_name)
    return slice_file


class Stage(enum.Enum):
    def __str__(self):
        return str(self.value)

    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"
    CONTROL = "cont"

def mlflow_create_experiment(experiment_name: str, default_tags = {"version": "v1", "priority": "P1"}) -> mlflow.entities.Experiment:
    # Create an experiment name, which must be unique and case sensitive
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    #FIXME : do logging instead of print 
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))
    return experiment