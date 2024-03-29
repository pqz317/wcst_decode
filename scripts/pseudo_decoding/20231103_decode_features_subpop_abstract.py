import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils

from utils.session_data import SessionData

from constants.decoding_constants import *

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_abstract_trial_splitter import ConditionAbstractTrialSplitter 
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
POSSIBLE_FEATURES = {
    "Color": ['CYAN', 'GREEN', 'MAGENTA', 'YELLOW'],
    "Shape": ['CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE'],
    "Pattern": ['ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL']
}

# the output directory to store the data
OUTPUT_DIR = "/data/res/pseudo"
# path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/{sess_name}_residual_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

# OUTPUT_DIR = "/data/patrick_res/pseudo"
# # path to a dataframe of sessions to analyze
# SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"

# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

DATA_MODE = "FiringRate"

# NOTE: this is important to use, especially for cross decoding. Should really have a better way to do this
def load_session_data(sess_name, condition, is_abstract, abs_cond, subpop, subtrials): 
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
    valid_beh_merged = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    valid_beh_merged = behavioral_utils.get_rpe_groups_per_session(sess_name, valid_beh_merged)

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
    if subpop is not None: 
        sess_subpop = subpop[subpop.session == sess_name]
        frs = frs[frs.UnitID.isin(sess_subpop.UnitID)]
        if len(frs) == 0:
            return None
    if subtrials is not None: 
        sess_subtrial = subtrials[subtrials.session == sess_name]
        valid_beh_merged = valid_beh_merged[valid_beh_merged.TrialNumber.isin(sess_subtrial.TrialNumber)]
        if len(valid_beh_merged) == 0: 
            return None
    if is_abstract:
        if not abs_cond: 
            split_conditions = [dim for dim in FEATURE_DIMS if not dim == condition]
        else: 
            split_conditions = [abs_cond]
        # create a trial splitter 
        splitter = ConditionAbstractTrialSplitter(valid_beh_merged, condition, split_conditions)
    else: 
        splitter = ConditionTrialSplitter(valid_beh_merged, condition, TEST_RATIO, seed=DECODER_SEED)
    sess_data = SessionData(sess_name, valid_beh_merged, frs, splitter)
    sess_data.pre_generate_splits(NUM_SPLITS)
    return sess_data

def decode_feature(feature_dim, valid_sess, is_abstract, abs_cond, subpop, subpop_name, subtrials, subtrials_name, proj, proj_name, l2_reg):
    """
    For a feature dimension and list of sessions, sets up and runs decoding, stores results
    Args: 
        feature_dim: feature dimension to decode
        valid_sess: a dataframe of valid sessions to be used
    """
    print(f"Decoding {feature_dim} with subpop {subpop_name} and is_abstract {is_abstract}")
    # load all session datas
    sess_datas = valid_sess.apply(lambda x: load_session_data(
        x.session_name, feature_dim, is_abstract, abs_cond, subpop, subtrials
    ), axis=1)
    sess_datas = sess_datas.dropna()
    
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = POSSIBLE_FEATURES[feature_dim]
    if proj is None:
        num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    else: 
        num_neurons = proj.shape[1]
    init_params = {"n_inputs": num_neurons, "p_dropout": P_DROPOUT, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=LEARNING_RATE, max_iter=MAX_ITER, weight_decay=l2_reg)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, DECODER_SEED, proj)

    if abs_cond:
        abstract_str = "abs_" + abs_cond
    elif is_abstract:
        abstract_str = "abs"
    else: 
        abstract_str = "baseline"

    # store the results
    # np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_train_accs.npy"), train_accs)
    # np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_test_accs.npy"), test_accs)
    # np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_shuffled_accs.npy"), shuffled_accs)
    # np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_models.npy"), models)

    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_residiual_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_residual_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_residual_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{feature_dim}_{abstract_str}_{subpop_name}_{subtrials_name}_{proj_name}_{l2_reg}_{DATA_MODE}_{INTERVAL_SIZE}_residual_models.npy"), models)

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--abstract', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--abs_cond', type=str, help="condition to be abstract respect to", default=None)
    parser.add_argument('--subpop_path', type=str, help="a path to subpopulation file", default="")
    parser.add_argument('--subpop_name', type=str, help="name of subpopulation", default="all")
    parser.add_argument('--subtrials_path', type=str, help="a path to subtrials file", default="")
    parser.add_argument('--subtrials_name', type=str, help="name of subtrials", default="all")
    parser.add_argument('--proj_path', type=str, help="a path to projection file", default="")
    parser.add_argument('--proj_name', type=str, help="a path to projection file", default="no_proj")
    parser.add_argument('--l2_reg', type=float, help="amount of l2 regularization", default=0.0)
    parser.add_argument('--feature_dim', type=str, help="feature dimension", default="")

    args = parser.parse_args()
    is_abstract = args.abstract
    abs_cond = args.abs_cond
    subpop_name = args.subpop_name
    subtrials_name = args.subtrials_name
    l2_reg = args.l2_reg
    if args.subpop_path:
        subpops = pd.read_pickle(args.subpop_path)
    else: 
        subpops = None
    if args.subtrials_path:
        subtrials = pd.read_pickle(args.subtrials_path)
    else: 
        subtrials = None
    proj_name = args.proj_name
    if args.proj_path:
        proj = np.load(args.proj_path)
    else: 
        proj = None
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    if args.feature_dim: 
        decode_feature(args.feature_dim, valid_sess, is_abstract, abs_cond, subpops, subpop_name, subtrials, subtrials_name, proj, proj_name, l2_reg)
    else: 
        for feature_dim in FEATURE_DIMS: 
            decode_feature(feature_dim, valid_sess, is_abstract, abs_cond, subpops, subpop_name, subtrials, subtrials_name, proj, proj_name, l2_reg)

if __name__ == "__main__":
    main()