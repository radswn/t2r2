import pandas as pd

from t2r2.selector import Selector


class ClSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["order"] = [len(i) for i in dataset["text"]]
        return dataset
