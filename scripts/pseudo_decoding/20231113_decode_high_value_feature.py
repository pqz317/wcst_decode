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

import argparse

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 50  # size of interval in ms

# all the possible feature dimensions 
# NOTE: Capital 1st letter is the convention here
FEATURE_DIMS = ["Color", "Shape", "Pattern"]
# for each feature dimension, list the possible classes
FEATURE_TO_DIM = {
    'CIRCLE': 'Shape', 
    'SQUARE': 'Shape', 
    'STAR': 'Shape', 
    'TRIANGLE': 'Shape', 
    'CYAN': 'Color', 
    'GREEN': 'Color', 
    'MAGENTA': 'Color', 
    'YELLOW': 'Color', 
    'ESCHER': 'Pattern', 
    'POLKADOT': 'Pattern', 
    'RIPPLE': 'Pattern', 
    'SWIRL': 'Pattern'
}
# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"
SESS_RESIDUAL_SPIKES_PATH = "/data/{sess_name}_residual_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

# DATA_MODE = "SpikeCounts"
DATA_MODE = "FiringRate"
TEST_RATIO = 0.2

SEED = 42

def load_session_data(sess_name, condition, subpops, subtrials, use_residual): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
    Returns: a SessionData object
    """
    model_path = f"/data/082023_Feat_RLDE_HV/sess-{sess_name}_hv.csv"
    model_vals = pd.read_csv(model_path)
    feat_names = np.array([
        'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
        'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
        'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
    ])
    renames = {}
    for i, feat_name in enumerate(feat_names):
        renames[f"feat_{i}"] = feat_name
    model_vals = model_vals.rename(columns=renames)

    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    if subtrials is not None: 
        sess_subtrial = subtrials[subtrials.session == sess_name]
        valid_beh_merged = valid_beh_merged[valid_beh_merged.TrialNumber.isin(sess_subtrial.TrialNumber)]
        if len(valid_beh_merged) == 0: 
            return None
    valid_beh_vals = pd.merge(valid_beh_merged, model_vals, left_on="TrialNumber", right_on="trial", how="inner")
    assert(len(valid_beh_vals) == len(valid_beh_merged))

    rng = np.random.default_rng(seed=SEED)
    def get_highest_val_feat(row):
        color = row["Color"]
        shape = row["Shape"]
        pattern = row["Pattern"]
        vals = {color: row[color], shape: row[shape], pattern: row[pattern]}
        max_feat = max(zip(vals.values(), vals.keys()))[1]
        random_feat = rng.choice(list(vals.keys()))
        row["MaxFeat"] = max_feat
        row["RandomMaxFeat"] = random_feat
        return row
    valid_beh_max = valid_beh_vals.apply(get_highest_val_feat, axis=1)

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
    if subpops is not None: 
        sess_subpop = subpops[subpops.session == sess_name]
        frs = frs[frs.UnitID.isin(sess_subpop.UnitID)]
        if len(frs) == 0:
            return None
    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_max, condition, TEST_RATIO, seed=SEED)
    session_data = SessionData(sess_name, valid_beh_max, frs, splitter)
    session_data.pre_generate_splits(8)
    return session_data

def decode_high_value(valid_sess, condition, subpops, subpop_name, subtrials, subrials_name, use_residual):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    # load all session datas
    sess_datas = valid_sess.apply(lambda x: load_session_data(x.session_name, condition, subpops, subtrials, use_residual), axis=1)
    sess_datas = sess_datas.dropna()

    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = list(FEATURE_TO_DIM.keys())
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
    residual_str = "residual_fr" if use_residual else "base_fr"

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"high_val_{condition}_{subpop_name}_{subrials_name}_{residual_str}_50_rpe_sess_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"high_val_{condition}_{subpop_name}_{subrials_name}_{residual_str}_50_rpe_sess_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"high_val_{condition}_{subpop_name}_{subrials_name}_{residual_str}_50_rpe_sess_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"high_val_{condition}_{subpop_name}_{subrials_name}_{residual_str}_50_rpe_sess_models.npy"), models)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default="MaxFeat")
    parser.add_argument('--subpop_path', type=str, help="a path to subpopulation file", default="")
    parser.add_argument('--subpop_name', type=str, help="name of subpopulation", default="all")
    parser.add_argument('--subtrials_path', type=str, help="a path to subtrials file", default="")
    parser.add_argument('--subtrials_name', type=str, help="name of subtrials", default="all")
    parser.add_argument('--use_residual_fr', action=argparse.BooleanOptionalAction, default=False)


    args = parser.parse_args()
    condition = args.condition
    subpop_name = args.subpop_name
    subtrials_name = args.subtrials_name

    if args.subpop_path:
        subpops = pd.read_pickle(args.subpop_path)
    else: 
        subpops = None
    if args.subtrials_path:
        subtrials = pd.read_pickle(args.subtrials_path)
    else: 
        subtrials = None

    valid_sess = pd.read_pickle(SESSIONS_PATH)
    decode_high_value(valid_sess, condition, subpops, subpop_name, subtrials, subtrials_name, args.use_residual_fr)

if __name__ == "__main__":
    main()