from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class ConditionKFoldBlockSplitter: 
    """
    Per condition specified, splits trials into train/test, ensuring that blocks 
    going into training are different from blocks going into testing. 
    Aims to ensure there is no temporal correlations exploitable 
    """

    def __init__(self, beh_df, condition_column, n_splits=5, block_splits=None, seed=None):
        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        if block_splits is None:
            kf = KFold(n_splits=n_splits, shuffle=True)
            blocks = beh_df.BlockNumber.unique()
            self.splits = kf.split(blocks)
        else:
            self.splits = block_splits
        self.block_split_iter = iter(self.splits)
        self.beh_df = beh_df
        trials = beh_df.groupby(by=condition_column).apply(lambda g: g.TrialNumber.unique())
        self.trials_df = pd.DataFrame({"Condition": trials.index, "TrialNumbers": trials.values})

    def __iter__(self):
        self.block_split_iter = iter(self.splits)
        return self

    def __next__(self):
        """
        Returns: 
            A dataframe with columns:
                - Condition: the specific condition of the row
                - TrainTrails: trials satisfying this condition for training, as a list per row
                - TestTrials: trials satisfying this condition held out for testing, as a list per row
        """
        def split_train_test(row, all_train_trials, all_test_trials):
            condition_trials = list(row["TrialNumbers"])
            row["TestTrials"] = np.intersect1d(condition_trials, all_train_trials)
            row["TrainTrials"] = np.intersect1d(condition_trials, all_test_trials)
            return row
        
        train_blocks, test_blocks = next(self.block_split_iter)
        all_train_trials = self.beh_df[self.beh_df.BlockNumber.isin(train_blocks)].TrialNumber.unique()
        all_test_trials = self.beh_df[self.beh_df.BlockNumber.isin(test_blocks)].TrialNumber.unique()

        trials_train_test = self.trials_df.apply(
            lambda row: split_train_test(row, all_train_trials, all_test_trials), 
            axis=1
        )
        return trials_train_test
