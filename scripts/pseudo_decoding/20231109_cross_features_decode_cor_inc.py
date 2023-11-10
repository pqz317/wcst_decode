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
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

import torch

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
# the output directory to store the data
# output dir that was specified in the decode_features_with_pseudo.py

OUTPUT_DIR = "/data/patrick_res/pseudo"
HYAK_OUTPUT_DIR = "/data/patrick_res/hyak/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/multi_sess/valid_sessions_rpe.pickle"
# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/multi_sess/{sess_name}/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

DATA_MODE = "SpikeCounts"

SEED = 42


def load_session_data(sess_name, condition): 
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

    # create a trial splitter 
    cor_splitter = ConditionTrialSplitter(cor_beh, condition, 0.2, seed=SEED)
    inc_splitter = ConditionTrialSplitter(inc_beh, condition, 0.2, seed=SEED)
    cor_data = SessionData(sess_name, cor_beh, frs, cor_splitter)
    inc_data = SessionData(sess_name, inc_beh, frs, inc_splitter)
    return (cor_data, inc_data)

def cross_decode_feature(feature_dim, sessions):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    print(f"Decoding {feature_dim}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    sess_datas = sessions.apply(lambda x: load_session_data(x.session_name, feature_dim), axis=1)
    cor_datas = sess_datas.apply(lambda x: x[0])
    inc_datas = sess_datas.apply(lambda x: x[1])
    cor_models = np.load(os.path.join(HYAK_OUTPUT_DIR, f"{feature_dim}_baseline_all_cor_models.npy"), allow_pickle=True)
    inc_models = np.load(os.path.join(HYAK_OUTPUT_DIR, f"{feature_dim}_baseline_all_inc_models.npy"), allow_pickle=True)
    time_bins = np.arange(0, 2.8, 0.1)
    cor_inc_accs = pseudo_classifier_utils.evaluate_model_with_data(cor_models, inc_datas, time_bins)
    inc_cor_accs = pseudo_classifier_utils.evaluate_model_with_data(inc_models, cor_datas, time_bins)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_cross_cor_models_inc_data_accs.npy"), cor_inc_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_cross_inc_models_cor_data_accs.npy"), inc_cor_accs)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    for feature_dim in FEATURE_DIMS: 
        cross_decode_feature(feature_dim, valid_sess)

if __name__ == "__main__":
    main()