import enum
import os
from collections.abc import MutableMapping
from typing import Union  # in later version of python simly replace with |

import pandas as pd


def flatten_dict(d: MutableMapping, sep: str = "_") -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


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
    VALIDATION = "validation"
    TESTING = "test"
    CONTROL = "control"


def check_if_directory_exists(output_path: str):
    """Checks if directory exists, if not creates it"""
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
