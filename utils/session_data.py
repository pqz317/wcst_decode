import numpy as np
import pandas as pd

from itertools import cycle

from . import pseudo_utils
from . import behavioral_utils
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter

PRE_INTERVAL = 1300
POST_INTERVAL = 1500
INTERVAL_SIZE = 100
EVENT = "FeedbackOnset"

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
            splitter: a TrialSplitter dictacting how 
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
        self.splitter = cycle(splits)
        return splits


    def generate_pseudo_data(self, num_train, num_test, time_bin):
        frs_at_bin = self.frs[np.isclose(self.frs.TimeBins, time_bin)]
        split = next(self.splitter)
        pseudo_pop = pseudo_utils.generate_pseudo_population(frs_at_bin, split, num_train, num_test)
        # print(self.sess_name)
        # print(len(pseudo_pop.PseudoTrialNumber.unique()))
        # print(pseudo_pop.Condition.unique())
        pseudo_pop["Session"] = self.sess_name
        # NOTE: very hacky way of giving unique ID to units across sessions
        pseudo_pop["PseudoUnitID"] = int(self.sess_name) * 100 + pseudo_pop["UnitID"]
        # print(len(pseudo_pop.PseudoUnitID.unique()))
        return pseudo_pop

    def get_num_neurons(self):
        return len(self.frs.UnitID.unique())

    
    @staticmethod
    def load_session_data(sess_name, condition): 
        behavior_path = f"/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
        beh = pd.read_csv(behavior_path)
        valid_beh = beh[beh.Response.isin(["Correct", "Incorrect"])]    
        feature_selections = behavioral_utils.get_selection_features(valid_beh)
        valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
        frs = pd.read_pickle(f"/data/patrick_scratch/multi_sess/{sess_name}/{sess_name}_firing_rates_{PRE_INTERVAL}_{EVENT}_{POST_INTERVAL}_{INTERVAL_SIZE}_bins.pickle")
        splitter = ConditionTrialSplitter(valid_beh_merged, condition, 0.2)
        return SessionData(sess_name, valid_beh_merged, frs, splitter)