"""
Metrics file
for slicing, needs input from slicing selector 
#FIXME: get config of overarching mechanisms needing both 
selectors and metrics (slicing) or metrics and training (data cartography)
"""
from snorkel.utils import preds_to_probs
from snorkel.analysis import Scorer
import pandas as pd
import os
from typing import Union
from enginora.utils import construct_results_file, Stage

#FIXME: later add config instead of passing all of those things


def slicing_scores(y_true, y_pred, slice_file: Union[str, None] = None, proba_predictions = None, base_directory=None, default_file_name = None, stage : Union[Stage, None] = None) -> pd.DataFrame:     
    slice_file = construct_results_file(slice_file, stage, base_directory, default_file_name)
    slices = pd.read_pickle(slice_file)
    scorer = Scorer(metrics=["accuracy"]) #3 add additional ones # no f1 for multiclass
    scorer_df = scorer.score_slices(
    S=slices[:len(y_true)], golds=y_true, preds=y_pred, probs=proba_predictions, as_dataframe=True)
    return scorer_df.to_json()



