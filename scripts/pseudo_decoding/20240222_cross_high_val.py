"""
Try to decode between two features, in cases when they're higher valued or not
Condition on correct trials
Be able to switch: 
- using residual firing rates or not
- whether group we're looking at is max or not
- feature pairs we're looking at
TODO: build cross decoder for these conditions
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

import argparse

# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SESS_RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


# # the output directory to store the data
# OUTPUT_DIR = "/data/patrick_res/pseudo"
# # path to a dataframe of sessions to analyze
# # SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"

# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


DATA_MODE = "FiringRate"
EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 50  # size of interval in ms

# FEATURE_DIM = "Shape"
# COND_TO_SPLIT = "MaxFeatMatches"
# CONDITIONS = [COND_TO_SPLIT, FEATURE_DIM]
# NUM_UNIQUE_CONDITIONS = 4
# FILTERS = {"Response": "Correct"}

# FEATURE_DIM = "Shape"
# CONDITIONS = [COND_TO_SPLIT, FEATURE_DIM]
NUM_UNIQUE_CONDITIONS = 8
FILTERS = {}
CONDITIONS = ["MaxFeat", "RandomMaxFeat"]
MIN_NUM_TRIALS = 20

SEED = 42

def load_session_data(sess_name, data_condition, use_residual): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
    Returns: a SessionData object
    """
    model_path = f"/data/082023_Feat_RLDE_HV/sess-{sess_name}_hv.csv"
    model_vals = pd.read_csv(model_path)
    feat_names = np.array([
        'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
        'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
        'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
    ])
    renames = {}
    for i, feat_name in enumerate(feat_names):
        renames[f"feat_{i}"] = feat_name
    model_vals = model_vals.rename(columns=renames)

    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    valid_beh_vals = pd.merge(valid_beh_merged, model_vals, left_on="TrialNumber", right_on="trial", how="inner")
    assert(len(valid_beh_vals) == len(valid_beh_merged))

    rng = np.random.default_rng(seed=SEED)
    def get_highest_val_feat(row):
        color = row["Color"]
        shape = row["Shape"]
        pattern = row["Pattern"]
        vals = {color: row[color], shape: row[shape], pattern: row[pattern]}
        max_feat = max(zip(vals.values(), vals.keys()))[1]
        random_feat = rng.choice(list(vals.keys()))
        row["MaxFeat"] = max_feat
        row["RandomMaxFeat"] = random_feat
        return row
    valid_beh_max = valid_beh_vals.apply(get_highest_val_feat, axis=1)

    # load firing rates
    if use_residual:
        format_path = SESS_RESIDUAL_SPIKES_PATH
    else: 
        format_path = SESS_SPIKES_PATH
    spikes_path = format_path.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})

    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_max, data_condition, TEST_RATIO, seed=SEED)
    session_data = SessionData(sess_name, valid_beh_max, frs, splitter)
    session_data.pre_generate_splits(8)
    return session_data

def decode(valid_sess, model_cond, use_residual):
    data_cond = [cond for cond in CONDITIONS if cond != model_cond][0]

    cond_other_sess_datas = valid_sess.apply(lambda row: load_session_data(row.session_name, data_cond, use_residual), axis=1)
    cond_other_sess_datas = cond_other_sess_datas.dropna()

    residual_str = "residual_fr" if use_residual else "base_fr"

    cond_model = np.load(
        os.path.join(OUTPUT_DIR, f"high_val_{model_cond}_all_all_{residual_str}_50_rpe_sess_models.npy"), 
        allow_pickle=True
    )
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    accs = pseudo_classifier_utils.evaluate_model_with_data(cond_model, cond_other_sess_datas, time_bins)

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"{model_cond}_model_on_{data_cond}_data_{residual_str}_cross_accs.npy"), accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default="all")
    parser.add_argument('--use_residual_fr', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    decode(valid_sess, args.condition, args.use_residual_fr)


if __name__ == "__main__":
    main()