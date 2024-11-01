"""
Evaluate CCGP of Value in neural population, computing belief state value from belief state agent
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
from tqdm import tqdm
import argparse

# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
PAIRS_PATH = "/data/pairs_at_least_3blocks_7sess.pickle"

# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# PAIRS_PATH = "/data/patrick_res/sessions/pairs_at_least_3blocks_10sess.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SIMULATED_SPIKES_PATH = "/data/firing_rates/{sess_name}_firing_rates_simulated_noise_{noise}.pickle"

DATA_MODE = "FiringRate"
EVENT = "StimOnset"  # event in behavior to align on
PRE_INTERVAL = 1000   # time in ms before event
POST_INTERVAL = 1000  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

REGIONS = ["anterior", "temporal"]
UNITS_PATH = "/data/patrick_res/firing_rates/all_units.pickle"

def load_session_data(row, cond, region_units):
    """
    cond: either a feature or a pair of features: 
    """
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_beliefs_per_session(beh, sess_name)
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
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if region_units: 
        frs["PseudoUnitID"] = int(sess_name) * 100 + int(frs.UnitID)
        frs = frs[frs.PseudoUnitID.isin(region_units)]
    splitter = ConditionTrialSplitter(sub_beh, "BeliefStateValueBin", TEST_RATIO)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data


def train_decoder(sess_datas):
    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = [0, 1]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND
    ) 
    return train_accs, test_accs, shuffled_accs, models

def get_region_units(region):
    if region is None: 
        return None
    all_units = pd.read_pickle(UNITS_PATH)
    if region == "anterior": 
        return all_units[all_units.Channel.str.contains('a')].PseudoUnitID.unique()
    elif region == "temporal":
        return all_units[~all_units.Channel.str.contains('a')].PseudoUnitID.unique()
    else: 
        raise ValueError(f"unrecognized region {region}")
    
def decode(sessions, row, region):
    pair = row.pair
    pair_str = "_".join(pair)
    within_cond_accs = []
    across_cond_accs = []
    region_str = "" if region is None else f"_{region}"
    name = f"ccgp_belief_state_value_{EVENT}_pair_{pair_str}{region_str}"
    region_units = get_region_units(region)
    for feat in pair: 
        print(f"Training decoder for low vs.  high {feat}")
        # load up session data to train network
        train_feat_sess_datas = sessions.apply(lambda row: load_session_data(row, [feat], region_units), axis=1)
        train_accs, test_accs, shuffled_accs, models = train_decoder(train_feat_sess_datas)
        within_cond_accs.append(test_accs)

        # next, evaluate network on other dimensions
        test_feat = [f for f in pair if f != feat][0]
        test_feat_sess_datas = sessions.apply(lambda row: load_session_data(row, [test_feat], region_units), axis=1)
        time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
        accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, test_feat_sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
        across_cond_accs.append(accs_across_time)
        np.save(os.path.join(OUTPUT_DIR, f"{name}_feat_{feat}_models.npy"), models)
    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)

    overall_sess_datas = sessions.apply(lambda row: load_session_data(row, pair, region_units), axis=1)
    train_accs, test_accs, shuffled_accs, models = train_decoder(overall_sess_datas)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_overall_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_overall_models.npy"), models)

    np.save(os.path.join(OUTPUT_DIR, f"{name}_within_cond_accs.npy"), within_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_across_cond_accs.npy"), across_cond_accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', default=None, type=int)
    parser.add_argument('--region_idx', default=None, type=str)
    args = parser.parse_args()
    pairs = pd.read_pickle(PAIRS_PATH)
    row = pairs.iloc[args.pair_idx]
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess = valid_sess[valid_sess.session_name.isin(row.sessions)]

    region =  None if args.region_idx is None else REGIONS[args.region_idx]

    print(f"Computing CCGP of belief state value between {row.pair} using between {row.num_sessions} sessions")
    decode(valid_sess, row, region)

if __name__ == "__main__":
    main()