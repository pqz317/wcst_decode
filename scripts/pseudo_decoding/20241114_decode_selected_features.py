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


EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions_rpe.pickle"

SESS_BEHAVIOR_PATH = "/data/patrick_res/behavior/{sub}/{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sub}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

DATA_MODE = "FiringRate"
REGIONS = ["anterior", "temporal", None]
UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"


def load_session_data(sess_name, feature_dim, subject, region_units, shuffle_idx): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
    Returns: a SessionData object
    """
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name, sub=subject)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    beh = behavioral_utils.get_valid_trials(beh, sub=subject)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(beh)
    beh = pd.merge(beh, feature_selections, on="TrialNumber", how="inner")

    if shuffle_idx is not None: 
        beh = behavioral_utils.shuffle_beh_by_shift(beh, buffer=50, seed=shuffle_idx)
    beh = behavioral_utils.balance_trials_by_condition(beh, feature_dim)


    # load firing rates
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
    # create a trial splitter 
    splitter = ConditionTrialSplitter(beh, feature_dim, TEST_RATIO, seed=DECODER_SEED)
    session_data = SessionData(sess_name, beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode_feature(sessions, feature_dim, subject, region, shuffle_idx):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    print(f"Decoding {feature_dim} for subject {subject}")
    region_str = "" if region is None else f"_{region}"
    shuffle_str = "" if shuffle_idx is None else f"_shuffle_{shuffle_idx}"
    name = f"{subject}_selected_features_{feature_dim}{region_str}{shuffle_str}"
    region_units = spike_utils.get_region_units(region, UNITS_PATH.format(sub=subject))
    # load all session datas
    sess_datas = sessions.apply(lambda x: load_session_data(x.session_name, feature_dim, subject, region_units, shuffle_idx), axis=1)

    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = POSSIBLE_FEATURES[feature_dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}

    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    trial_interval = get_trial_interval(EVENT)
    time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND
    ) 

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"{name}_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_models.npy"), models)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--region_idx', default=None, type=int)
    parser.add_argument('--feature_dim_idx', type=int)
    parser.add_argument('--region_idx', default=None, type=int)
    parser.add_argument('--subject', default="SA", type=str)
    parser.add_argument('--shuffle_idx', default=None, type=int)
    args = parser.parse_args()

    sessions = pd.read_pickle(SESSIONS_PATH.format(sub=args.subject))
    # args.region =  None if args.region_idx is None else REGIONS[args.region_idx]
    feature_dim = FEATURE_DIMS[args.feature_dim_idx]
    region = None if args.region_idx is None else REGIONS[args.region_idx]
    decode_feature(sessions, feature_dim, args.subject, region, args.shuffle_idx)

if __name__ == "__main__":
    main()