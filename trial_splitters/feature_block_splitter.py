from __future__ import annotations
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
import pandas as pd
import numpy as np

class FeatureBlockSplitter(TrialSplitter): 
    """Splits trials into train/test sets by block
    On each iteration, holds one block out as the test set, the rest
    trials as the training set.
    """

    def __init__(self, df: pd.Series) -> None:
        self.df = df
        self.rule_color = df[df.CurrentRule.isin(['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'])]
        self.rule_shape = df[df.CurrentRule.isin(['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'])]
        self.rule_pattern = df[df.CurrentRule.isin(['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'])]
        self.blocks_color = self.rule_color.BlockNumber.unique()
        self.blocks_shape = self.rule_shape.BlockNumber.unique()
        self.blocks_pattern = self.rule_pattern.BlockNumber.unique()
        self.min_len = np.min((len(self.blocks_color), len(self.blocks_shape), len(self.blocks_pattern)))


    def __iter__(self) -> FeatureBlockSplitter:
        self.block_idx = 0
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        if self.block_idx < self.min_len:
            test_blocks = [
                self.blocks_color[self.block_idx], 
                self.blocks_pattern[self.block_idx], 
                self.blocks_shape[self.block_idx]
            ]
            test = self.df[self.df.BlockNumber.isin(test_blocks)].TrialNumber.unique()
            train = self.df[~self.df.TrialNumber.isin(test)].TrialNumber.unique()
            self.block_idx += 1
            return (train, test)
        raise StopIteration

    def __len__(self) -> int:
        return self.min_len
