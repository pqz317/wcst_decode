from __future__ import annotations
from typing import Tuple, List
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from .trial_splitter import TrialSplitter
import pandas as pd
import numpy as np




class RuleConditionBlockSplitter: 
    """
    Per rule, splits trials into train/test by leaving one block out for testing, and letting the rest be training

    """

    def __init__(self, beh_df, seed=None):
        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        self.beh_df = beh_df
        blocks = beh_df.groupby(by="CurrentRule").apply(lambda g: g.BlockNumber.unique())
        self.blocks_df = pd.DataFrame({"Condition": blocks.index, "Blocks": blocks.values})

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns: 
            A dataframe with columns:
                - Condition: the specific condition of the row, in this case the current rule
                - TrainTrails: trials satisfying this condition for training, as a list per row
                - TestTrials: trials satisfying this condition held out for testing, as a list per row
        """
        def split_train_test(row):
            blocks = list(row["Blocks"])
            self.rng.shuffle(blocks)
            row["TestBlocks"] = blocks[:1]
            row["TrainBlocks"] = blocks[1:]
            row["TestTrials"] = self.beh_df[self.beh_df["BlockNumber"].isin(row["TestBlocks"])].TrialNumber.values
            row["TrainTrials"] = self.beh_df[self.beh_df["BlockNumber"].isin(row["TrainBlocks"])].TrialNumber.values
            return row
        blocks_train_test = self.blocks_df.apply(lambda row: split_train_test(row), axis=1)
        return blocks_train_test
