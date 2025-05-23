from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class ConditionKFoldSplitter: 
    """
    Per condition specified, splits trials into train/test with a kfold splitter. 
    """

    def __init__(self, beh_df, condition_column, num_splits, min_trials_per_cond=2, seed=None):
        """
        Args: 
            beh_df: behavioral dataframe, with condition column, and TrialNumbers column
            condition_column: column used to split on. train/test splits will be assigned per-condition
            test_ratio: ratio of test trials to split out
            min_trials_per_cond: minimum number of trials for each condition. 
                Will raise an error during iteration if not satisfied. 
        """
        # get condition -> block numbers
        self.seed = seed
        self.beh_df = beh_df
        self.min_trials_per_cond = min_trials_per_cond
        self.num_splits = num_splits

        trials = beh_df.groupby(by=condition_column).apply(lambda g: g.TrialNumber.unique())
        self.trials_df = pd.DataFrame({"Condition": trials.index, "TrialNumbers": trials.values})
        self._check_min_trials()
        self.splits = self.create_splits()
        self.index = 0

    def create_splits(self):
        """
        creates splits by going through each condition, and splitting the trials into kfold splits
        """
        splits = [[] for _ in range(self.num_splits)]
        kf = KFold(n_splits=self.num_splits, random_state=self.seed, shuffle=True)
        def split_train_test(row):
            trials = row["TrialNumbers"]
            for i, (train_idxs, test_idxs) in enumerate(kf.split(trials)):
                split_row = {}
                split_row["Condition"] = row.Condition
                split_row["TrainTrials"] = trials[train_idxs]
                split_row["TestTrials"] = trials[test_idxs]
                splits[i].append(split_row)
        self.trials_df.apply(lambda row: split_train_test(row), axis=1)
        return [pd.DataFrame(split) for split in splits]
        


    def _check_min_trials(self):
        def check_cond(row):
            trials = list(row["TrialNumbers"]) 
            if len(trials) < self.min_trials_per_cond:
                raise ValueError(f"Condition {row.Condition} only has {len(trials)} trials, need {self.min_trials_per_cond}")
        self.trials_df.apply(check_cond, axis=1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.num_splits:
            raise StopIteration
        split = self.splits[self.index]
        self.index += 1
        return split