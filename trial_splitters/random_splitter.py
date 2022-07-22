from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter

class RandomSplitter(TrialSplitter):
    """Splits trials into train/test sets randomly on each iteration"""
    def __init__(self, trial_numbers: List[int], num_runs: int, test_size: float) -> None:
        self.trial_numbers = trial_numbers
        self.num_runs = num_runs
        self.test_size = test_size

    def __iter__(self) -> RandomSplitter:
        self.n = 0
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        if self.n < self.num_runs:
            train, test = train_test_split(self.trial_numbers, test_size=self.test_size)
            self.n += 1
            return (train, test)
        raise StopIteration
