import pandas as pd
from enginora.selector import Selector

class UserSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[:5]
