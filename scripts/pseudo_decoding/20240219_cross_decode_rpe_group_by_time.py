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
import argparse

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 50  # size of interval in ms

# # the output directory to store the data
# OUTPUT_DIR = "/data/patrick_res/pseudo"
# # path to a dataframe of sessions to analyze
# # SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


# DATA_MODE = "SpikeCounts"
DATA_MODE = "FiringRate"
TEST_RATIO = 0.2

FEATURE_DIMS = ["Color", "Shape", "Pattern"]

# NOTE: should match whichever seed was used to generate splits for the models
SEED = 42
NUM_SPLITS = 8

def load_session_data(sess_name, subpops): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
    Returns: a SessionData object
    """
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")

    valid_beh_rpes = behavioral_utils.get_rpe_groups_per_session(sess_name, valid_beh)

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
    if subpops is not None: 
        sess_subpop = subpops[subpops.session == sess_name]
        frs = frs[frs.UnitID.isin(sess_subpop.UnitID)]
        if len(frs) == 0:
            return None
    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_rpes, "RPEGroup", 0.2, seed=SEED)
    sess_data = SessionData(sess_name, valid_beh_rpes, frs, splitter)
    sess_data.pre_generate_splits(NUM_SPLITS)
    return sess_data


def decode(valid_sess, subpops):
    print(f"Cross decoding RPE Group") 
    sess_datas = valid_sess.apply(lambda x: load_session_data(x.session_name, subpops), axis=1)
    sess_datas = sess_datas.dropna()

    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # models = np.load(os.path.join(OUTPUT_DIR, f"{feature_dim}_baseline_{subpop_name}_all_no_proj_0.0_models.npy"), allow_pickle=True)
    models = np.load(os.path.join(OUTPUT_DIR, f"rpe_groups_all_no_proj_all_FiringRate_50_models.npy"), allow_pickle=True)

    cross_decode_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, time_bins, avg=False)
    np.save(os.path.join(OUTPUT_DIR, f"rpe_groups_cross_acc_FiringRate_50.npy"), cross_decode_accs)

def main(subpops, subpop_name):
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: remove: 
    print(device)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    decode(valid_sess, subpops)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--subpop_path', type=str, help="a path to subpopulation file", default="")
    parser.add_argument('--subpop_name', type=str, help="name of subpopulation", default="all")

    args = parser.parse_args()
    subpop_name = args.subpop_name
    if args.subpop_path:
        subpops = pd.read_pickle(args.subpop_path)
    else: 
        subpops = None

    main(subpops, subpop_name)