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

    def __init__(self, beh_df, condition_column, test_ratio, min_trials_per_cond=2, seed=None):
        """
        Args: 
            beh_df: behavioral dataframe, with condition column, and TrialNumbers column
            condition_column: column used to split on. train/test splits will be assigned per-condition
            test_ratio: ratio of test trials to split out
            min_trials_per_cond: minimum number of trials for each condition. 
                Will raise an error during iteration if not satisfied. 
        """
        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        self.beh_df = beh_df
        self.min_trials_per_cond = min_trials_per_cond
        trials = beh_df.groupby(by=condition_column).apply(lambda g: g.TrialNumber.unique())
        self.trials_df = pd.DataFrame({"Condition": trials.index, "TrialNumbers": trials.values})
        self._check_min_trials()
        self.test_ratio = test_ratio

    def _check_min_trials(self):
        def check_cond(row):
            trials = list(row["TrialNumbers"]) 
            if len(trials) < self.min_trials_per_cond:
                raise ValueError(f"Condition {row.Condition} only has {len(trials)} trials, need {self.min_trials_per_cond}")
        self.trials_df.apply(check_cond, axis=1)

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
            # round up number of test trials
            # ex. if test ratio at 0.2, even with 2 trials, train/test split will still be 1/1. 
            split_at = np.ceil(len(trials) * self.test_ratio).astype(int)
            row["TestTrials"] = trials[:split_at]
            row["TrainTrials"] = trials[split_at:]
            return row
        trials_train_test = self.trials_df.apply(lambda row: split_train_test(row), axis=1)
        return trials_train_test
