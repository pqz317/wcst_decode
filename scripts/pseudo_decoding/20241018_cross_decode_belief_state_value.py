"""
Cross decode belief state value by time. 
Probably interested in cross decoding for overall, as well as within condition. 
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils

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
# OUTPUT_DIR = "/data/patrick_res/hyak/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sub}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SA_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_7sess.pickle"
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"


DATA_MODE = "FiringRate"
# EVENT = "StimOnset"  # event in behavior to align on
# PRE_INTERVAL = 1000   # time in ms before event
# POST_INTERVAL = 1000  # time in ms after event
# INTERVAL_SIZE = 100  # size of interval in ms

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

REGIONS = ["anterior", "temporal"]

def load_session_data(row, cond, region_units, subject):
    """
    cond: either a feature or a pair of features: 
    """
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name, sub=subject)
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_valid_trials(beh, sub=subject)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, sess_name, subject)
    beh = behavioral_utils.get_belief_value_labels(beh)

    # subselect for either low conf, or high conf preferring feat, where feat is also chosen
    if len(cond) == 1:
        feat = cond[0] 
        sub_beh = beh[
            ((beh[FEATURE_TO_DIM[feat]] == feat) & (beh.BeliefStateValueLabel == f"High {feat}")) |
            (beh.BeliefStateValueLabel == "Low")
        ]
    elif len(cond) == 2: 
        feat1, feat2 = cond
        sub_beh = beh[
            ((beh[FEATURE_TO_DIM[feat1]] == feat1) & (beh.BeliefStateValueLabel == f"High {feat1}")) |
            ((beh[FEATURE_TO_DIM[feat2]] == feat2) & (beh.BeliefStateValueLabel == f"High {feat2}")) |
            (beh.BeliefStateValueLabel == "Low")
        ]
    else: 
        raise ValueError("cond must be either 1 or 2 elements")

    # balance the conditions out: 
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["BeliefStateValueBin"])
    spikes_path = SESS_SPIKES_PATH.format(
        sub=subject,
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if region_units is not None: 
        frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
        frs = frs[frs.PseudoUnitID.isin(region_units)]
    splitter = ConditionTrialSplitter(sub_beh, "BeliefStateValueBin", TEST_RATIO)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(sessions, pair_row, region, subject="SA"):
    pair = pair_row.pair
    pair_str = "_".join(pair)
    region_str = "" if region is None else f"_{region}"
    region_units = spike_utils.get_region_units(region, UNITS_PATH.format(sub=subject))

    name = f"{subject}_ccgp_belief_state_value_{EVENT}_pair_{pair_str}{region_str}"
    # for feat in pair: 
    #     sess_datas = sessions.apply(lambda row: load_session_data(row, [feat], region_units, subject), axis=1)
    #     time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    #     models = np.load(os.path.join(OUTPUT_DIR, f"{name}_feat_{feat}_models.npy"), allow_pickle=True)
    #     cross_decode_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, time_bins, avg=False)
    #     np.save(os.path.join(OUTPUT_DIR, f"{name}_feat_{feat}_cross_accs.npy"), cross_decode_accs)
    
    sess_datas = sessions.apply(lambda row: load_session_data(row, pair, region_units, subject), axis=1)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    models = np.load(os.path.join(OUTPUT_DIR, f"{name}_overall_models.npy"), allow_pickle=True)
    cross_decode_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, time_bins, avg=False)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_overall_cross_accs.npy"), cross_decode_accs)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', default=None, type=int)
    parser.add_argument('--region_idx', default=None, type=int)
    args = parser.parse_args()
    pairs = pd.read_pickle(SA_PAIRS_PATH)
    row = pairs.iloc[args.pair_idx]
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub="SA"))
    valid_sess = valid_sess[valid_sess.session_name.isin(row.sessions)]

    region =  None if args.region_idx is None else REGIONS[args.region_idx]

    print(f"Cross decoding belief state value by time, looking at pair {row.pair} between {row.num_sessions} sessions, region {region}")
    decode(valid_sess, row, region)

if __name__ == "__main__":
    main()