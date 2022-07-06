from __future__ import annotations
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from trial_splitters.trial_splitter import TrialSplitter
import pandas as pd


class GroupSplitter(TrialSplitter): 
    """
    Splits trials into train/test sets by group
    On each iteration, holds one group out as the test set, the rest 
    trials as the training set. 
    """

    def __init__(self, trials_with_groups: pd.DataFrame) -> None:
        self.trials_with_groups = trials_with_groups
        self.groups = trials_with_groups.group.unique()

    def __iter__(self) -> GroupSplitter:
        self.group_idx = 0
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        if self.group_idx < len(self.groups):
            group = self.groups[self.group_idx]
            test = self.trials_with_groups[self.trials_with_groups["group"] == group].TrialNumber.unique()
            train = self.trials_with_groups[self.trials_with_groups["group"] != group].TrialNumber.unique()
            self.group_idx += 1
            return (test, train)
        else: 
            raise StopIteration
