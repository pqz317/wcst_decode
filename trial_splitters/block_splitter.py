from __future__ import annotations
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
import pandas as pd

class BlockSplitter(TrialSplitter): 
    """Splits trials into train/test sets by block
    On each iteration, holds one block out as the test set, the rest
    trials as the training set.
    """

    def __init__(self, trials_with_blocks: pd.Series) -> None:
        self.trials_with_blocks = trials_with_blocks
        self.blocks = trials_with_blocks.BlockNumber.unique()

    def __iter__(self) -> BlockSplitter:
        self.block_idx = 0
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        if self.block_idx < len(self.blocks):
            block = self.blocks[self.block_idx]
            test = self.trials_with_blocks[self.trials_with_blocks["BlockNumber"] == block].TrialNumber.unique()
            train = self.trials_with_blocks[self.trials_with_blocks["BlockNumber"] != block].TrialNumber.unique()
            self.block_idx += 1
            return (train, test)
        raise StopIteration
