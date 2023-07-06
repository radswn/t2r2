"""
Module for implementing slicing functions by snorkel.
"""
# FIXME: is this really a good module tree?
from enginora.selector.base import *
from snorkel.analysis import Scorer
from snorkel.slicing import SlicingFunction, slicing_function
from snorkel.slicing import PandasSFApplier
from typing import Union, List, Callable
import pandas as pd
from enginora.selector.slicing import default_slicing_functions
from enginora.selector.slicing.default_slicing_functions import short, textblob_polarity

# default slicing functions for text


class SlicingSelector(Selector):
    def __init__(self, result_file: str, list_of_slicing_functions=None):
        super().__init__()
        self.sfs = self.create_list_of_slicing_functions(list_of_slicing_functions)
        self.result_file = result_file

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # TODO: thinking about slicing in general: I highly doubt that is is truly a selector...
        if not len(self.sfs) == 0:
            res = self.create_slicing_functions(dataset)
            res.dump(self.result_file)
        return dataset

    def create_list_of_slicing_functions(self, list_of_slicing_functions: Union[List[str], None]):
        if list_of_slicing_functions is None:
            sfs = [short]
        else:
            sfs = [getattr(default_slicing_functions, function_name) for function_name in list_of_slicing_functions]
        return sfs

    def create_slicing_functions(self, dataset):
        applier = PandasSFApplier(self.sfs)
        slice_membership = applier.apply(dataset)
        return slice_membership
