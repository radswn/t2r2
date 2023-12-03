from abc import ABC, abstractmethod

import pandas as pd


class Selector(ABC):
    """Abstract class for selectors"""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Method to be overwritten by individual selectors"""
