"""
Module for implementing slicing functions by snorkel.
"""
# FIXME: is this really a good module tree?
from enginora.selector.base import *
from snorkel.analysis import Scorer
from textblob import TextBlob
from snorkel.slicing import SlicingFunction, slicing_function
from snorkel.preprocess import preprocessor
from snorkel.slicing import PandasSFApplier
from typing import Union, List, Callable


@slicing_function()
def short(x):
    return len(x.text.split()) < 5


@preprocessor(memoize=True)
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    return x


@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x):
    return x.polarity > 0.9


class SlicingSelector(Selector):
    # wether or not to create default
    def __init__(self, result_file: str, list_of_slicing_functions=None):
        super().__init__()
        self.sfs = self.create_list_of_slicing_functions(list_of_slicing_functions)
        self.result_file = result_file

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        res = self.create_slicing_functions(dataset)
        return res

    def create_list_of_slicing_functions(
        self, list_of_slicing_functions: Union[List[Callable], None]
    ):
        # FIXME: add functionality not to rely on default (TEXT)
        if list_of_slicing_functions is None:
            sfs = [short, textblob_polarity]
        else:
            sfs = list_of_slicing_functions
        return sfs

    def create_slicing_functions(self, dataset):
        applier = PandasSFApplier(self.sfs)
        res = applier.apply(dataset)  # FIXME: name nicely
        return res
