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

PRE_INTERVAL = 300   # time in ms before event
POST_INTERVAL = 500  # time in ms after event
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
OUTPUT_DIR = "/data/patrick_res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/patrick_res/multi_sess/valid_sessions_rpe.pickle"

# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/multi_sess/{sess_name}/{sess_name}_firing_rates_{pre_interval}_fixation_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SESS_FIXATIONS_PATH = "/data/patrick_res/multi_sess/{sess_name}/{sess_name}_fixations.pickle"

DATA_MODE = "SpikeCounts"

TEST_RATIO = 0.2

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
    fixation_path = SESS_FIXATIONS_PATH.format(sess_name=sess_name)
    fixations = pd.read_pickle(fixation_path)

    # load firing rates
    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})

    # create a trial splitter 
    splitter = ConditionTrialSplitter(fixations, condition, TEST_RATIO, seed=SEED)
    session_data = SessionData(sess_name, fixations, frs, splitter)
    session_data.pre_generate_splits(8)
    return session_data

def decode_feature(feature_dim, valid_sess):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    print(f"Decoding {feature_dim}")
    # load all session datas
    sess_datas = valid_sess.apply(lambda x: load_session_data(x.session_name, feature_dim), axis=1)

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

    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_fixations_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_fixations_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_fixations_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_fixations_models.npy"), models)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    for feature_dim in FEATURE_DIMS: 
        decode_feature(feature_dim, valid_sess)

if __name__ == "__main__":
    main()