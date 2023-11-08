from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
import pandas as pd
import numpy as np

class ConditionAbstractTrialSplitter: 
    """
    Per condition specified, splits trials into train/test randomly. 
    """

    def __init__(self, beh_df, decode_column, split_columns):
        # get condition -> block numbers
        self.beh_df = beh_df
        self.splits = []
        self.decode_column = decode_column
        for split_column in split_columns:
            splits = [(split_column, val) for val in beh_df[split_column].unique()]
            self.splits = self.splits + splits
        self.splits_iter = iter(self.splits)

    def __iter__(self):
        self.splits_iter = iter(self.splits)
        return self

    def __next__(self):
        """
        Returns: 
            A dataframe with columns:
                - Condition: the specific condition of the row
                - TrainTrails: trials satisfying this condition for training, as a list per row
                - TestTrials: trials satisfying this condition held out for testing, as a list per row
        """
        cur_column, cur_val = next(self.splits_iter)
        def split_train_test(group):
            row = {}
            row["TestTrials"] = group[group[cur_column] != cur_val].TrialNumber.unique()
            row["TrainTrials"] = group[group[cur_column] == cur_val].TrialNumber.unique()
            return pd.Series(row)
        trials_train_test = self.beh_df.groupby(self.decode_column).apply(split_train_test).reset_index()
        trials_train_test = trials_train_test.rename(columns={self.decode_column: "Condition"})
        return trials_train_test
