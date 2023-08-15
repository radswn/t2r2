import inspect
import functools
from typing import Union  # in later version of python simly replace with |
import os
import enum
from collections.abc import MutableMapping
import pandas as pd


def flatten_dict(d: MutableMapping, sep: str = "_") -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


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
