from enginora.selector.base import *


class DummySelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
