from enginora.selector.base import *
from typing import Union

import numpy as np


class DataCartographySelector(Selector):
    def __init__(self, hard_to_learn: float = 0.0,
                 easy_to_learn: float = 0.0,
                 ambiguous: float = 0.0,
                 random: float = 0.0,
                 input_filepath: str = "./data_cartography_metrics.pickle",
                 random_state: Union[int, None] = None
                 ):
        assert hard_to_learn + easy_to_learn + ambiguous + random <= 1
        assert hard_to_learn > 0 or easy_to_learn > 0 or ambiguous > 0
        super().__init__()
        self.hard_to_learn = hard_to_learn
        self.easy_to_learn = easy_to_learn
        self.ambiguous = ambiguous
        self.random = random
        self.input_filepath = input_filepath
        self.random_state = random_state if random_state is not None else np.random.RandomState.randint(0)

    def read_input(self):
        return pd.read_pickle(self.input_filepath)

    def select_right_indices(self) -> list:
        df = self.read_input()
        size_of_new_dataset = len(df)

        easy_sample_size = int(self.easy_to_learn * size_of_new_dataset)
        hard_sample_size = int(self.hard_to_learn * size_of_new_dataset)
        ambiguous_sample_size = int(self.ambiguous * size_of_new_dataset)

        # Sort the DataFrame by "variability" and "correctness" in descending order
        df_sorted_variability = df.sort_values(by='variability', ascending=False)
        df_sorted_correctness = df.sort_values(by='correctness', ascending=False)

        # Select the top 50% of records with the highest variability

        easy_indices = df_sorted_variability.index[:ambiguous_sample_size] if easy_sample_size else []

        # Select the bottom 10% of records with the lowest correctness
        hard_indices = df_sorted_correctness.index[-hard_sample_size:] if hard_sample_size else []

        # Select the top 30% of records with the highest correctness
        ambiguous_indices = df_sorted_correctness.index[:easy_sample_size] if ambiguous_sample_size else []

        random_indices = df.sample(frac=self.random, replace=False, random_state=self.random_satate).index if self.random > 0 else []
        # Combine the selected indices into a set (without duplicates)
        selected_indices = list(set(easy_indices) | set(hard_indices) | set(ambiguous_indices) | set(random_indices))
        return selected_indices

    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        indices = self.select_right_indices()
        return dataset.iloc[indices]


