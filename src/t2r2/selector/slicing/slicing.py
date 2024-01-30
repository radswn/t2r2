"""
Module for implementing slicing functions by snorkel.
"""

from typing import List, Union

import pandas as pd
from snorkel.slicing import PandasSFApplier

from t2r2.selector.base import Selector
from t2r2.selector.slicing import default_slicing_functions
from t2r2.selector.slicing.default_slicing_functions import short, long
from t2r2.utils import check_if_directory_exists


class SlicingSelector(Selector):
    def __init__(self, result_file: str, list_of_slicing_functions=None, **kwargs):
        super().__init__()
        self.sfs = self.create_list_of_slicing_functions(list_of_slicing_functions)
        self.result_file = result_file

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        check_if_directory_exists(self.result_file)
        if not len(self.sfs) == 0:
            res = self.create_slicing_functions(dataset)
            res.dump(self.result_file)
        return dataset

    def create_list_of_slicing_functions(self, list_of_slicing_functions: Union[List[str], None]):
        if list_of_slicing_functions is None:
            sfs = [short, long]
        else:
            sfs = [getattr(default_slicing_functions, function_name) for function_name in list_of_slicing_functions]
        return sfs

    def create_slicing_functions(self, dataset):
        applier = PandasSFApplier(self.sfs)
        slice_membership = applier.apply(dataset)
        return slice_membership
