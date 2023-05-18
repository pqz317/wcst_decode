from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import KFold
from .trial_splitter import TrialSplitter
import pandas as pd


# Define a custom block splitter cause the only one works with trials, not row idxs
class KFoldBlockSplitter: 
    """Splits trials into train/test sets by block
    On each iteration, holds one block out as the test set, the rest
    trials as the training set.
    """

    def __init__(self, trials_with_blocks: pd.Series, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True)
        self.n_splits = n_splits
        self.trials_with_blocks = trials_with_blocks
        self.blocks = trials_with_blocks.BlockNumber.unique()
        self.splits = kf.split(self.blocks)
        self.iter = iter(self.splits)

    def __iter__(self):
        self.iter = iter(self.splits)
        return self

    def __next__(self):
        train_idxs, test_idxs = next(self.iter)

        test = self.trials_with_blocks[self.trials_with_blocks["BlockNumber"].isin(self.blocks[train_idxs])].TrialNumber.unique()
        train = self.trials_with_blocks[self.trials_with_blocks["BlockNumber"].isin(self.blocks[test_idxs])].TrialNumber.unique()
        return (train, test)

    def __len__(self):
        return self.n_splits