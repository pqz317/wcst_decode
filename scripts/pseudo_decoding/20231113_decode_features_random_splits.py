import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_abstract_trial_splitter import ConditionAbstractTrialSplitter 
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter
import argparse

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

# all the possible feature dimensions 
# NOTE: Capital 1st letter is the convention here
FEATURE_DIMS = ["Color", "Shape", "Pattern"]
# for each feature dimension, list the possible classes
POSSIBLE_FEATURES = {
    "Color": ['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'],
    "Shape": ['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'],
    "Pattern": ['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL']
}

OUTPUT_DIR = "/data/patrick_res/pseudo"
HYAK_OUTPUT_DIR = "/data/patrick_res/hyak/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


DATA_MODE = "SpikeCounts"

TEST_RATIO = 0.2

def load_session_data(sess_name, condition, shuffle_idx): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
        condition: condition used to group trials in pseudo population (in this case a feature dimension)
    Returns: a SessionData object
    """
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")

    cor_beh = valid_beh[valid_beh.Response == "Correct"]
    inc_beh = valid_beh[valid_beh.Response == "Incorrect"]
    min_num_trials = np.min((len(cor_beh), len(inc_beh)))

    trials = valid_beh.TrialNumber.values.copy()
    rng = np.random.default_rng(seed=shuffle_idx)
    rng.shuffle(trials)
    # get first N and last N trials
    group_A_trials = trials[:min_num_trials]
    group_B_trials = trials[-min_num_trials:]
    beh_A = valid_beh[valid_beh.TrialNumber.isin(group_A_trials)]
    beh_B = valid_beh[valid_beh.TrialNumber.isin(group_B_trials)]

    splitter_A = ConditionTrialSplitter(beh_A, condition, TEST_RATIO)
    splitter_B = ConditionTrialSplitter(beh_B, condition, TEST_RATIO)

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
 
    sess_data_A = SessionData(sess_name, beh_A, frs, splitter_A)
    sess_data_A.pre_generate_splits(8)

    sess_data_B = SessionData(sess_name, beh_B, frs, splitter_B)
    sess_data_B.pre_generate_splits(8)

    return (sess_data_A, sess_data_B)

def decode_for_group(feature_dim, sess_datas, group, shuffle_idx):
    print(f"Decoding {feature_dim}")
    
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = POSSIBLE_FEATURES[feature_dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()

    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(model, sess_datas, time_bins, 8, 2000, 500, 42)

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_group_{group}_shuffle_{shuffle_idx}_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_group_{group}_shuffle_{shuffle_idx}__test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_group_{group}_shuffle_{shuffle_idx}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_group_{group}_shuffle_{shuffle_idx}_models.npy"), models)


def decode_feature(feature_dim, valid_sess, shuffle_idx):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    sess_datas = valid_sess.apply(lambda x: load_session_data(
        x.session_name, feature_dim, shuffle_idx
    ), axis=1)
    sess_datas = sess_datas.dropna()
    group_A_datas = sess_datas.apply(lambda x: x[0])
    group_B_datas = sess_datas.apply(lambda x: x[1])
    decode_for_group(feature_dim, group_A_datas, "A", shuffle_idx)
    decode_for_group(feature_dim, group_B_datas, "B", shuffle_idx)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('shuffle_idx', type=int, help="int from 0 - 10 denoting which session to run for")

    args = parser.parse_args()
    shuffle_idx = int(args.shuffle_idx)
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    for feature_dim in FEATURE_DIMS: 
        decode_feature(feature_dim, valid_sess, shuffle_idx)

if __name__ == "__main__":
    main()