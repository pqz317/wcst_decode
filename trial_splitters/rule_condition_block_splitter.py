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
    IMPORTANT: condition here should only be specified as either CurrentRulr or RuleDim. We want to ensure that
    train/test sets don't bleed in to each other
    """

    def __init__(self, beh_df, condition="CurrentRule", seed=None, num_distinct_conditions=12):
        # get condition -> block numbers
        self.rng = np.random.default_rng(seed=seed)
        self.beh_df = beh_df
        blocks = beh_df.groupby(by=condition).apply(lambda g: g.BlockNumber.unique())
        self.blocks_df = pd.DataFrame({"Condition": blocks.index, "Blocks": blocks.values})
        if len(self.blocks_df) != num_distinct_conditions:
            raise ValueError(f"not the right number of conditions, with: {self.blocks_df.Condition.unique()}")
        # verify that each condition has at least 2 blocks
        self.blocks_df["NumBlocks"] = self.blocks_df.apply(lambda x: len(x.Blocks), axis=1)
        less_than_twos = self.blocks_df[self.blocks_df.NumBlocks < 2]
        if len(less_than_twos) > 0:
            raise ValueError(f"conditions {less_than_twos.Condition.unique()} have less than two associated blocks")


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
