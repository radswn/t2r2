from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pandas as pd


class Selector(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


@dataclass
class SelectorConfig:
    name: str
    args: Dict
