from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class ConditionKFoldBlockSplitter: 
    def __init__(self, beh_df, condition_column, n_splits=5, block_splits=None, seed=None, min_trials_per_cond=1, num_distinct_conditions=4):
        """
        Per condition specified, splits trials into train/test, ensuring that blocks 
        going into training are different from blocks going into testing. 
        Aims to ensure there is no temporal correlations exploitable         
        Args: 
            beh_df: dataframe of trials, must contain the condition_column as one of the columns
            condition_column: condition to create the splitter around
            n_splits: number of splits to create, default 5
            block_splits: optional, if specified, creates splits based on train/test blocks passed in. 
            seed: optional, seeds rng
            min_trials_per_cond: minimum number of trials per condition in train/test sets. 
        """

        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        if block_splits is None:
            # kfold gives back indices instead of values...
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            blocks = beh_df.BlockNumber.unique()
            block_split_gen = kf.split(blocks)
            self.splits = [(blocks[train], blocks[test]) for (train, test) in block_split_gen]
        else:
            self.splits = block_splits
        self.block_split_iter = iter(self.splits)
        self.beh_df = beh_df
        trials = beh_df.groupby(by=condition_column).apply(lambda g: g.TrialNumber.unique())
        self.trials_df = pd.DataFrame({"Condition": trials.index, "TrialNumbers": trials.values})
        if len(self.trials_df) != num_distinct_conditions:
            raise ValueError(f"not the right number of conditions, with {self.trials_df.Condition.unique()}")
        self.min_trials_per_cond = min_trials_per_cond

    def __iter__(self):
        self.block_split_iter = iter(self.splits)
        return self

    def __next__(self):
        """
        Creates train/test sets for conditions based on block splits
        Throws: 
            ValueError if not enough trials in train/test for each condition
        Returns: 
            A dataframe with columns:
                - Condition: the specific condition of the row
                - TrainTrails: trials satisfying this condition for training, as a list per row
                - TestTrials: trials satisfying this condition held out for testing, as a list per row
        """
        def split_train_test(row, train_blocks, test_blocks):
            condition_trials = list(row["TrialNumbers"])
            all_train_trials = self.beh_df[self.beh_df.BlockNumber.isin(train_blocks)].TrialNumber.unique()
            all_test_trials = self.beh_df[self.beh_df.BlockNumber.isin(test_blocks)].TrialNumber.unique()

            train_trials = np.intersect1d(condition_trials, all_train_trials)
            if len(train_trials) < self.min_trials_per_cond: 
                raise ValueError(f"There are not enough trials ({len(train_trials)}) for condition {row.Condition} in train blocks {train_blocks}")
            test_trials = np.intersect1d(condition_trials, all_test_trials)
            if len(test_trials) < self.min_trials_per_cond:
                raise ValueError(f"There are not enough trials ({len(test_trials)}) for condition {row.Condition} in test blocks {test_blocks}")
        
            row["TrainTrials"] = train_trials
            row["TestTrials"] = test_trials
            return row
        
        train_blocks, test_blocks = next(self.block_split_iter)

        trials_train_test = self.trials_df.apply(
            lambda row: split_train_test(row, train_blocks, test_blocks), 
            axis=1
        )
        return trials_train_test
