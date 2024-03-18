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


DATA_MODE = "FiringRate"
EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 50  # size of interval in ms

# # FEATURE_DIM = "Shape"
# COND_TO_SPLIT = "MaxFeatMatches"
# # CONDITIONS = [COND_TO_SPLIT, FEATURE_DIM]
# NUM_UNIQUE_CONDITIONS = 4
# FILTERS = {"Response": "Correct"}

# FEATURE_DIM = "Shape"
COND_TO_SPLIT = "Response"
# CONDITIONS = [COND_TO_SPLIT, FEATURE_DIM]
NUM_UNIQUE_CONDITIONS = 8
FILTERS = {}

MIN_NUM_TRIALS = 20


def get_feat_beh(session, feat, shuffle):
    feat_beh = behavioral_utils.get_beh_model_labels_for_session_feat(session, feat, beh_path=SESS_BEHAVIOR_PATH)
    if shuffle:
        rng = np.random.default_rng(seed=DECODER_SEED)
        vals = feat_beh[COND_TO_SPLIT].values
        rng.shuffle(vals)
        feat_beh[COND_TO_SPLIT] = vals
    return feat_beh


def label_and_balance_sessions(session, features, feature_dim, shuffle):
    feat_behs = []
    for feat in features:
        feat_behs.append(get_feat_beh(session, feat, shuffle))
    beh = pd.concat(feat_behs)
    # subselect for correct 
    for filter_col, filter in FILTERS.items():
        beh = beh[beh[filter_col] == filter]
    conditions_cols = [COND_TO_SPLIT, feature_dim]
    enough_trials = behavioral_utils.validate_enough_trials_by_condition(
        beh, 
        conditions_cols, 
        MIN_NUM_TRIALS, 
        num_unique_conditions=NUM_UNIQUE_CONDITIONS
    )
    if not enough_trials:
        print("Not enough trials for session {session}, skipping")
        return None
    balanced_beh = behavioral_utils.balance_trials_by_condition(beh, conditions_cols)
    return balanced_beh

def load_session_data(sess_group, feature_dim, use_residual):
    sess_name = sess_group.Session.iloc[0]
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
    splitter = ConditionTrialSplitter(sess_group, feature_dim, TEST_RATIO, seed=DECODER_SEED)
    session_data = SessionData(sess_name, sess_group, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(all_trials, features, feature_dim, condition, use_residual, should_shuffle):
    if condition == "all":
        cond_all_trials = all_trials
    else: 
        cond_all_trials = all_trials[all_trials[COND_TO_SPLIT].astype(str) == condition]

    sess_datas = cond_all_trials.groupby("Session").apply(lambda group: load_session_data(group, feature_dim, use_residual))
    classes = features
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": P_DROPOUT, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=LEARNING_RATE, max_iter=MAX_ITER)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, DECODER_SEED
    )

    # store the results
    residual_str = "residual_feature_fb_fr" if use_residual else "base_fr"
    shuffle_str = "shuffled" if should_shuffle else "unshuffled"
    features_str = "_vs_".join(features)
    np.save(os.path.join(OUTPUT_DIR, f"{features_str}_{COND_TO_SPLIT}_{condition}_{residual_str}_{shuffle_str}_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{features_str}_{COND_TO_SPLIT}_{condition}_{residual_str}_{shuffle_str}_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{features_str}_{COND_TO_SPLIT}_{condition}_{residual_str}_{shuffle_str}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{features_str}_{COND_TO_SPLIT}_{condition}_{residual_str}_{shuffle_str}_models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_list', default="SQUARE,TRIANGLE", type=str, help="comma separated list of features")
    parser.add_argument('--feature_dim', default="Shape", type=str)
    parser.add_argument('--condition', default="all")
    parser.add_argument('--use_residual_fr', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--should_shuffle', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    features = args.feature_list.split(",")
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    balanced_sessions = valid_sess.apply(lambda x: label_and_balance_sessions(
        x.session_name, 
        features,
        args.feature_dim,
        args.should_shuffle,
    ), axis=1)
    balanced_sessions = balanced_sessions.dropna()
    print(f"Decoding between {', '.join(features)} using {len(balanced_sessions)} sessions")

    balanced_all_trials = pd.concat(balanced_sessions.values)
    decode(balanced_all_trials, features, args.feature_dim, args.condition, args.use_residual_fr, args.should_shuffle)


if __name__ == "__main__":
    main()