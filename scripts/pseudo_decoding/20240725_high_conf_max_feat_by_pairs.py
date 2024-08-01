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
PAIRS_PATH = "/data/pairs_at_least_3blocks_7sess.pickle"
MIN_TRIALS_FOR_PAIRS_PATH = "/data/pairs_at_least_3blocks_7sess_min_trials.pickle"

# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# PAIRS_PATH = "/data/patrick_res/sessions/pairs_at_least_3blocks_10sess.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


DATA_MODE = "FiringRate"
EVENT = "StimOnset"  # event in behavior to align on
PRE_INTERVAL = 1000   # time in ms before event
POST_INTERVAL = 1000  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms
# EVENT = "FeedbackOnset"  # event in behavior to align on
# PRE_INTERVAL = 1300   # time in ms before event
# POST_INTERVAL = 1500  # time in ms after event
# INTERVAL_SIZE = 100  # size of interval in ms

def load_session_data(row, pair, shuffle_idx=None, seed_idx=None):
    feat1, feat2 = pair
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

    # shift TrialNumbers by some random amount
    if shuffle_idx is not None: 
        beh = behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=shuffle_idx)

    sub_beh = beh[beh["ConfidenceLabel"].isin([f"High {feat1}", f"High {feat2}"])]


    # balance the conditions out:
    # use minimum number of trials stored for the session/pair
    min_trials = pd.read_pickle(MIN_TRIALS_FOR_PAIRS_PATH) 
    min_num_trials = min_trials[
        (min_trials.pair.isin([pair])) & 
        (min_trials.session == sess_name)
    ].iloc[0].min_all
    sub_beh = behavioral_utils.balance_trials_by_condition(sub_beh, ["ConfidenceLabel"], seed=seed_idx, min=min_num_trials)

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(sub_beh, "ConfidenceLabel", TEST_RATIO, seed=seed_idx)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(sessions, row, shuffle_idx=None):
    pair = row.pair
    pair_str = "_".join(pair)

    # load up session data to train network
    sess_datas = sessions.apply(lambda row: load_session_data(row, pair, shuffle_idx), axis=1)

    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = [f"High {pair[0]}", f"High {pair[1]}"]
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

    shuffle_str = f"shuffle_{shuffle_idx}_" if shuffle_idx is not None else ""
    run_name = f"high_conf_fixed_max_feat_by_pairs_{EVENT}_pair_{pair_str}_{shuffle_str}"

    np.save(os.path.join(OUTPUT_DIR, f"{run_name}train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_idx', default=None, type=int)
    parser.add_argument('--shuffle_idx', default=None, type=int)

    args = parser.parse_args()
    pairs = pd.read_pickle(PAIRS_PATH)
    row = pairs.iloc[args.pair_idx]
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    valid_sess = valid_sess[valid_sess.session_name.isin(row.sessions)]

    print(f"Decoding between {row.pair} using between {row.num_sessions} sessions")
    decode(valid_sess, row, args.shuffle_idx)


if __name__ == "__main__":
    main()