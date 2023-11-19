import os
import numpy as np
import pandas as pd
import utils.behavioral_utils as behavioral_utils
import utils.information_utils as information_utils
import utils.visualization_utils as visualization_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils

import utils.glm_utils as glm_utils
from matplotlib import pyplot as plt
import utils.spike_utils as spike_utils
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 
from utils.session_data import SessionData

import torch

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

DATA_MODE = "SpikeCounts"
TEST_RATIO = 0.2

FEATURE_DIMS = ["Color", "Shape", "Pattern"]

def load_session_data(sess_name, condition): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
        condition: condition used to group trials in pseudo population (in this case a feature dimension)
    Returns: a SessionData object
    """
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")

    # load firing rates
    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})

    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_merged, condition, TEST_RATIO, seed=42)
    session_data = SessionData(sess_name, valid_beh_merged, frs, splitter)
    session_data.pre_generate_splits(8)
    return session_data

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    for feature_dim in FEATURE_DIMS:
        print(f"Cross decoding {feature_dim}") 
        sess_datas = valid_sess.apply(lambda x: load_session_data(x.session_name, feature_dim), axis=1)
        input_bins = np.arange(0, 2.8, 0.1)
        models = np.load(os.path.join(OUTPUT_DIR, f"{feature_dim}_rpe_sess_models.npy"), allow_pickle=True)
        cross_decode_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, input_bins, avg=False)
        np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_rpe_sess_cross_acc_alls.npy"), cross_decode_accs)

if __name__ == "__main__":
    main()