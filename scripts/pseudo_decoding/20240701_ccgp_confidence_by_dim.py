"""
Evauate abstractness of confidence by computing CCGP, cross condition generalization performance
Do this across dimensions
Filter for: 
- when attended to features (highest valued in behavior model) also match chosen card. 
Conditions are: 
- attended to feature belongs in some feature dimension
For each script, pass in seed idx
For each session, compute session data, min number of trials that match
To create session_data, For each dimension: 
    - 
    - sample N trials per session. 
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
SESS_BEHAVIOR_PATH = "/data/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


DATA_MODE = "FiringRate"
# EVENT = "StimOnset"  # event in behavior to align on
# PRE_INTERVAL = 1000   # time in ms before event
# POST_INTERVAL = 1000  # time in ms after event
# INTERVAL_SIZE = 100  # size of interval in ms
EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

def load_session_data(row, dim, seed_idx=None, use_next_trial_confidence=False):
    sess_name = row.session_name

    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_feature_values_per_session(sess_name, valid_beh)
    beh = behavioral_utils.get_max_feature_value(beh)
    beh = behavioral_utils.calc_feature_probs(beh)
    beh = behavioral_utils.calc_feature_value_entropy(beh)
    beh = behavioral_utils.calc_confidence(beh, num_bins=2, quantize_bins=True)
    if use_next_trial_confidence:
        beh["ConfidenceBin"] = beh["ConfidenceBin"].shift(-1)
        beh = beh[~beh["ConfidenceBin"].isna()]
        beh["ConfidenceBin"] = beh["ConfidenceBin"].astype(int)
        # filter by max chosen, also by dimension of interest
    beh = behavioral_utils.filter_max_feat_chosen(beh)

    # balance the conditions out: 
    beh = behavioral_utils.balance_trials_by_condition(beh, ["MaxFeatDim", "ConfidenceBin"], seed=seed_idx)

    # subselect only dimension specified
    beh = beh[beh.MaxFeatDim == dim]

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(beh, "ConfidenceBin", TEST_RATIO, seed=seed_idx)
    session_data = SessionData(sess_name, beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(sessions, seed_idx, use_next_trial_conf):
    within_cond_accs = []
    across_cond_accs = []
    next_trial_str = "next_trial_conf_" if use_next_trial_conf else ""

    for dim in FEATURE_DIMS:
        # load up session data to train network
        sess_datas = sessions.apply(lambda row: load_session_data(row, dim, seed_idx, use_next_trial_conf), axis=1)

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
            model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, seed_idx
        ) 
        within_cond_accs.append(test_accs)
        # test accs from network are within-condition accuracies

        # next, evaluate network on other dimensions
        for other_dim in FEATURE_DIMS:
            if other_dim == dim: 
                continue
            sess_datas = sessions.apply(lambda row: load_session_data(row, other_dim, seed_idx, use_next_trial_conf), axis=1)
            accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
            across_cond_accs.append(accs_across_time)
        np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{EVENT}_{seed_idx}_{next_trial_str}{dim}_models.npy"), models)

    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{EVENT}_{seed_idx}_{next_trial_str}within_dim_accs.npy"), within_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{EVENT}_{seed_idx}_{next_trial_str}across_dim_accs.npy"), across_cond_accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_idx', default=None, type=int)
    parser.add_argument('--use_next_trial_conf', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    print(f"Decoding confidence using {len(valid_sess)} sessions")
    decode(valid_sess, args.seed_idx, args.use_next_trial_conf)


if __name__ == "__main__":
    main()