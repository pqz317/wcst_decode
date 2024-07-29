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
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"
PAIRS_PATH = "/data/pairs_at_least_3blocks_10sess.pickle"

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
# EVENT = "FeedbackOnset"  # event in behavior to align on
# PRE_INTERVAL = 1300   # time in ms before event
# POST_INTERVAL = 1500  # time in ms after event
# INTERVAL_SIZE = 100  # size of interval in ms

SIM_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]

def load_session_data(row, feat, seed_idx=None, sim_noise=None):
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
    beh["ConfidenceLabel"] = beh.apply(lambda row: f"High {row.MaxFeat}" if row.ConfidenceBin == 1 else "Low", axis=1)
    sub_beh = beh[beh["ConfidenceLabel"].isin([f"High {feat}", "Low"])]


    # balance the conditions out: 
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["ConfidenceBin"], seed=seed_idx)
    if sim_noise is None: 
        spikes_path = SESS_SPIKES_PATH.format(
            sess_name=sess_name, 
            pre_interval=PRE_INTERVAL, 
            event=EVENT, 
            post_interval=POST_INTERVAL, 
            interval_size=INTERVAL_SIZE
        )
    else: 
        spikes_path = SIMULATED_SPIKES_PATH.format(
            sess_name=sess_name, 
            noise=sim_noise,
        )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(sub_beh, "ConfidenceBin", TEST_RATIO, seed=seed_idx)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(sessions, row, sim_noise=None):
    pair = row.pair
    pair_str = "_".join(pair)
    event_str = EVENT if sim_noise is None else f"simulation_noise_{sim_noise}"
    within_cond_accs = []
    across_cond_accs = []

    for feat in pair:
        # load up session data to train network
        sess_datas = sessions.apply(lambda row: load_session_data(row, feat, sim_noise=sim_noise), axis=1)

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
        within_cond_accs.append(test_accs)

        # test accs from network are within-condition accuracies

        # next, evaluate network on other dimensions
        other_feat = [f for f in pair if f != feat][0]
        sess_datas = sessions.apply(lambda row: load_session_data(row, other_feat, sim_noise=sim_noise), axis=1)
        accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
        across_cond_accs.append(accs_across_time)

        np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{event_str}_pair_{pair_str}_feat_{feat}_models.npy"), models)

    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{event_str}_pair_{pair_str}_within_dim_accs.npy"), within_cond_accs)
    np.save(os.path.join(OUTPUT_DIR, f"ccgp_confidence_{event_str}_pair_{pair_str}_across_dim_accs.npy"), across_cond_accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', default=None, type=int)
    parser.add_argument('--noise_idx', default=None, type=int)
    args = parser.parse_args()
    pairs = pd.read_pickle(PAIRS_PATH)
    row = pairs.iloc[args.pair_idx]
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess = valid_sess[valid_sess.session_name.isin(row.sessions)]
    sim_noise = None if args.noise_idx is None else SIM_NOISE_LEVELS[args.noise_idx]
    print(f"Computing CCGP of conf between {row.pair} using between {row.num_sessions} sessions")
    decode(valid_sess, row, sim_noise)


if __name__ == "__main__":
    main()