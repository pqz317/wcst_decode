import numpy as np
import pandas as pd

from itertools import cycle

from . import pseudo_utils
from . import behavioral_utils

class SessionData: 
    """
    A class to keep track of relevant session data for pseudo population generation
    """
    def __init__(self, sess_name, beh, frs, splitter):
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
        self.splitter = splitter

    def pre_generate_splits(self, num_splits):
        """
        Generates splits ahead of time so that each call of generate_pseudo_data
        uses the same num_splits number of splits
        Args: 
            num_splits: the number of splits to generate
        Returns:
            the generate splits to save 
        Modifies: 
            changes self.splitter into a cycle iterator of already generated splits
        """
        
        # generate a list of splits num_splits times
        splits = [next(self.splitter) for _ in range(num_splits)]
        # make is so that an iterator cycles through this list
        # if at each time bin, pseudo data is generated the same num_splits times, t
        # this ensures each time bin comes from the same set of splits
        self.splits = splits
        self.splitter = cycle(splits)
        return splits


    def generate_pseudo_data(self, num_train, num_test, time_bin, use_v2=False):
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
        split = next(self.splitter)
        # TODO: Change when testing
        if use_v2: 
            pseudo_pop = pseudo_utils.generate_pseudo_population_2(frs_at_bin, split, num_train, num_test)
        else: 
            pseudo_pop = pseudo_utils.generate_pseudo_population(frs_at_bin, split, num_train, num_test)

        pseudo_pop["Session"] = self.sess_name
        # NOTE: very hacky way of giving unique ID to units across sessions
        pseudo_pop["PseudoUnitID"] = int(self.sess_name) * 100 + pseudo_pop["UnitID"]
        # print(pseudo_pop[:5])
        # print(pseudo_pop.PseudoUnitID.unique())
        return pseudo_pop

    def get_num_neurons(self):
        return len(self.frs.UnitID.unique())
    
    def get_pseudo_unit_ids(self):
        return self.frs["PseudoUnitID"].unique()