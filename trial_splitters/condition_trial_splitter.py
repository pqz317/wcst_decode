from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
import pandas as pd
import numpy as np

class ConditionTrialSplitter: 
    """
    Per condition specified, splits trials into train/test randomly. 
    """

    def __init__(self, beh_df, condition_column, test_ratio, seed=None):
        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        self.beh_df = beh_df
        trials = beh_df.groupby(by=condition_column).apply(lambda g: g.TrialNumber.unique())
        self.trials_df = pd.DataFrame({"Condition": trials.index, "TrialNumbers": trials.values})
        self.test_ratio = test_ratio

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns: 
            A dataframe with columns:
                - Condition: the specific condition of the row
                - TrainTrails: trials satisfying this condition for training, as a list per row
                - TestTrials: trials satisfying this condition held out for testing, as a list per row
        """
        def split_train_test(row):
            trials = list(row["TrialNumbers"])
            self.rng.shuffle(trials)
            split_at = int(len(trials) * self.test_ratio)
            row["TestTrials"] = trials[:split_at]
            row["TrainTrials"] = trials[split_at:]
            return row
        trials_train_test = self.trials_df.apply(lambda row: split_train_test(row), axis=1)
        return trials_train_test
