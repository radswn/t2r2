"""
Metrics file
for slicing, needs input from slicing selector 
#FIXME: get config of overarching mechanisms needing both 
selectors and metrics (slicing) or metrics and training (data cartography)
"""
import os
from typing import Union

import pandas as pd
from snorkel.analysis import Scorer

from t2r2.utils import Stage, construct_results_file

# FIXME: later add config instead of passing all of those things


def slicing_scores(
    y_true,
    y_pred,
    slice_file: Union[str, None] = None,
    proba_predictions=None,
    base_directory=None,
    default_file_name=None,
    stage: Union[Stage, None] = None,
    metrics=["accuracy"],
) -> pd.DataFrame:
    slice_file = construct_results_file(slice_file, stage, base_directory, default_file_name)

    if os.path.isfile(slice_file):
        slices = pd.read_pickle(slice_file)
        scorer = Scorer(metrics=metrics)

        scorer_df = scorer.score_slices(
            S=slices[: len(y_true)], golds=y_true, preds=y_pred, probs=proba_predictions, as_dataframe=True
        )
    else:
        scorer_df = pd.DataFrame()

    return scorer_df.to_dict()
