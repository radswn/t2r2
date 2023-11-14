from t2r2.selector.base import *


class DummySelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset
