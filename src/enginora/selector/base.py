from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pandas as pd


class Selector(ABC):
    """Abstract class for selectors"""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Method to be overwritten by individual selectors"""


@dataclass
class SelectorConfig:
    """Abstract class for selectorCongigs for training, testing and control"""

    name: str
    args: Dict
