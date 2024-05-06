# Try to decode the feature with the maximum value
# during the inter-trial interval, just use aggregated spike counts

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
import time


# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
# SESS_RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SESS_RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_feature_fb_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


# # the output directory to store the data
# OUTPUT_DIR = "/data/patrick_res/pseudo"
# # path to a dataframe of sessions to analyze
# # SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"

# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


PRE_INTERVAL = 500
POST_INTERVAL = 500
INTERVAL_SIZE = 50
NUM_BINS_SMOOTH = 1
EVENT = "FixationOnCross"

DATA_MODE = "FiringRate"

def load_session_data(row, should_shuffle=False, shuffle_seed=None):
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_feature_values_per_session(sess_name, valid_beh)
    beh = behavioral_utils.get_max_feature_value(beh)
    if should_shuffle:
        trial_numbers = beh.TrialNumber.copy()
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(trial_numbers)
        beh["TrialNumber"] = trial_numbers

    num_trials_per_feat = beh.groupby("MaxFeat").TrialNumber.nunique()
    num_feats = len(num_trials_per_feat)
    if num_feats < 12 or np.any(num_trials_per_feat.values < 5):
        return None

    # load firing rates
    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()
    # hacky, but just pretend there's one timebin. 
    frs["TimeBins"] = 0
    frs = frs.rename(columns={DATA_MODE: "Value"})


    splitter = ConditionTrialSplitter(beh, "MaxFeat", TEST_RATIO, seed=DECODER_SEED)
    session_data = SessionData(sess_name, beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(valid_sess, should_shuffle, shuffle_seed):
    sess_datas = valid_sess.apply(load_session_data, axis=1)
    sess_datas = sess_datas.dropna()
    print(f"decoding from {len(sess_datas)} sessions")

    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": P_DROPOUT, "n_classes": 12}
    # create a trainer object
    trainer = Trainer(learning_rate=LEARNING_RATE, max_iter=MAX_ITER)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, FEATURES)
    # calculate time bins (in seconds)
    time_bins = np.zeros(1)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, DECODER_SEED
    )
    shuffle_str = ""
    if should_shuffle:
        shuffle_str = "shuffle_" if shuffle_seed is None else f"shuffle_{shuffle_seed}_"

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"intertrial_agg_max_feat_{shuffle_str}train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"intertrial_agg_max_feat_{shuffle_str}test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"intertrial_agg_max_feat_{shuffle_str}shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"intertrial_agg_max_feat_{shuffle_str}models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--should_shuffle', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--shuffle_seed', type=int, default=None)

    args = parser.parse_args()
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    decode(valid_sess, args.should_shuffle, args.shuffle_seed)
    end = time.time()
    print(f"Decoding took {(end - start) / 60} minutes")


if __name__ == "__main__":
    main()