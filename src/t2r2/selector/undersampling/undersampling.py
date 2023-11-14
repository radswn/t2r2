from imblearn.under_sampling import RandomUnderSampler

from t2r2.selector.base import *


class RandomUnderSamplerSelector(Selector):
    def __init__(self, **kwargs):
        super().__init__()
        self.sampler = RandomUnderSampler(**kwargs)

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return self.sampler.fit_resample(dataset, dataset["label"])[0]
