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
SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"

# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/multi_sess_deprecated/{sess_name}/{sess_name}_firing_rates_{pre_interval}_fixation_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SESS_FIXATIONS_PATH = "/data/patrick_res/multi_sess_deprecated/{sess_name}/{sess_name}_fixations.pickle"

DATA_MODE = "SpikeCounts"

TEST_RATIO = 0.5

SEED = 42

def load_session_data(sess_name, condition): 
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

    input_bins = np.arange(0, 0.8, 0.1)
    models = np.load(os.path.join(OUTPUT_DIR, f"{feature_dim}_rpe_sess_models.npy"), allow_pickle=True)
    cross_decode_accs = pseudo_classifier_utils.cross_evaluate_by_time_bins(models, sess_datas, input_bins)

    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_selection_fixations_cross_accs.npy"), cross_decode_accs)


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