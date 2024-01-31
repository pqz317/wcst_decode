"""
Asks whether decoding ability of a feature being the max value 
or not is modulated by confidence
"""

import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
from utils.constants import *

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

import argparse

# # the output directory to store the data
# OUTPUT_DIR = "/data/res/pseudo"
# # path to a dataframe of sessions to analyze
# # SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
# SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


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
EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

TEST_RATIO = 0.2
SEED = 42
MIN_TRIALS_PER_COND = 50

def balance_sessions(session, feat):
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=session)

    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    feat_dim = FEATURE_TO_DIM[feat]
    valid_beh_merged = valid_beh_merged[valid_beh_merged[feat_dim] == feat]
    valid_beh_vals = behavioral_utils.get_feature_values_per_session(session, valid_beh_merged)
    valid_beh_vals_conf = behavioral_utils.get_rpes_per_session(session, valid_beh_vals)
    med_conf = np.median(valid_beh_vals_conf["Prob_FE"].to_numpy())
    def assign_conf(row, med):
        row["Conf"] = "high" if row["Prob_FE"] > med else "low"
        return row
    valid_beh_vals_conf = valid_beh_vals_conf.apply(lambda row: assign_conf(row, med_conf), axis=1)
    valid_beh_vals_conf["MaxFeatMatches"] = valid_beh_vals_conf.MaxFeat == feat
    valid_beh_vals_conf["Session"] = session
    
    conditions = ["MaxFeatMatches", "Conf"]
    is_valid_sess = behavioral_utils.validate_enough_trials_by_condition(valid_beh_vals_conf, conditions, MIN_TRIALS_PER_COND)
    if not is_valid_sess:
        return None
    balanced = behavioral_utils.balance_trials_by_condition(valid_beh_vals_conf, conditions)
    return balanced

def load_session_data(sess_group):
    sess_name = sess_group.Session.iloc[0]
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
    splitter = ConditionTrialSplitter(sess_group, "MaxFeatMatches", TEST_RATIO, seed=SEED)
    session_data = SessionData(sess_name, sess_group, frs, splitter)
    session_data.pre_generate_splits(8)
    return session_data

def decode_max(all_trials, conf, feat):
    conf_all_trials = all_trials[all_trials.Conf == conf]
    sess_datas = conf_all_trials.groupby("Session").apply(load_session_data)
    classes = [True, False]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 8, 1000, 250, 42)

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"MaxFeat_{feat}_{conf}_conf_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"MaxFeat_{feat}_{conf}_conf_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"MaxFeat_{feat}_{conf}_conf_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"MaxFeat_{feat}_{conf}_conf_models.npy"), models)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', default="")
    parser.add_argument('--subpop_path', type=str, help="a path to subpopulation file", default="")
    parser.add_argument('--subpop_name', type=str, help="name of subpopulation", default="all")

    args = parser.parse_args()
    subpop_name = args.subpop_name
    feature = args.feature
    if args.subpop_path:
        subpops = pd.read_pickle(args.subpop_path)
    else: 
        subpops = None

    valid_sess = pd.read_pickle(SESSIONS_PATH)
    balanced_sessions = valid_sess.apply(lambda x: balance_sessions(x.session_name, feature), axis=1)
    balanced_sessions = balanced_sessions.dropna()
    print(f"Decoding {feature} trials using {len(balanced_sessions)} sessions")

    balanced_all_trials = pd.concat(balanced_sessions.values)

    print(f"Decoding low confidence trials")
    decode_max(balanced_all_trials, "low", feature)

    print(f"Decoding high confidence trials")
    decode_max(balanced_all_trials, "high", feature)


if __name__ == "__main__":
    main()