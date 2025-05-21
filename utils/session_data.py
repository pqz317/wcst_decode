import numpy as np
import pandas as pd

from itertools import cycle

from . import pseudo_utils
from . import behavioral_utils

class SessionData: 
    """
    A class to keep track of relevant session data for pseudo population generation
    """
    def __init__(self, sess_name, beh, frs, splits_df):
        """
        Creates a SessionData object
        Args: 
            sess_name: an str name for the session, ex: 20180802
            beh: behavior dataframe 
            frs: dataframe of firing rates, spike counts, by unit, time_bin
            splitter: dictates how to split trials by condition, in order to generate pseudo trials
        """
        self.sess_name = sess_name
        self.beh = beh
        self.frs = frs
        self.splits_df = splits_df

    def generate_pseudo_data(self, num_train, num_test, time_bin, split_idx):
        """
        For a specified timebin, generate num_train and num_test pseudotrials per condition
        With the existing trialsplitter
        Args: 
            num_train: number of train pseudo trials per condition to generate
            num_test: number of test pseudo trials per condition to generate
            time_bin: specific time_bin to generate for
        Returns: 
            A pseudo population dataframe specific to this session with columns:
                - Session: session name
                - PseudoUnitID: identifier for unit across sessions
                - UnitID: ID of unit (neuron) selected
                - TrialNumber: ID of original trial selected from
                - PseudoTrialNumber: ID of the pseudo trial generated
                - Type: Either Train or Test 
                - Condition
                - TimeBins
                - some data column (eg. SpikeCounts or FiringRates)
        """
        frs_at_bin = self.frs[np.isclose(self.frs.TimeBins, time_bin)]
        split = self.splits_df[self.splits_df.split_idx == split_idx]

        pseudo_pop = pseudo_utils.generate_pseudo_population_v2(frs_at_bin, split, num_train, num_test)
        pseudo_pop["Session"] = self.sess_name
        # NOTE: very hacky way of giving unique ID to units across sessions
        pseudo_pop["PseudoUnitID"] = int(self.sess_name) * 100 + pseudo_pop["UnitID"]
        return pseudo_pop
    
    def get_num_neurons(self):
        return len(self.frs.UnitID.unique())
    
    def get_pseudo_unit_ids(self):
        return self.frs["PseudoUnitID"].unique()
    
    def get_splits_df(self):
        """
        Converts the array of splits into a dataframe, adds a column for session name. 
        """
        return self.splits_df
    

def create_from_splitter(sess_name, beh, frs, splitter, num_splits):
    splits = []
    for split_idx in range(num_splits):
        split = next(splitter)
        split["split_idx"] = split_idx
        splits.append(split)
    splits_df = pd.concat(splits)
    splits_df["session"] = sess_name
    return SessionData(sess_name, beh, frs, splits_df)

def create_from_splits_df(sess_name, beh, frs, splits_df):
    sess_splits_df = splits_df[splits_df.session == sess_name]
    return SessionData(sess_name, beh, frs, sess_splits_df)