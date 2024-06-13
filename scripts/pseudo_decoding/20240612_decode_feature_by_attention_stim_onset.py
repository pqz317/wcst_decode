"""
Decode features by dimension
Split by condition of whether features are high valued or not
Subselect trials where high valued feature is also chosen
Look during simulus onset, 500ms before, 500ms after. 
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
SESSIONS_PATH = "/data/valid_sessions_by_dim.pickle"

# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"


DATA_MODE = "FiringRate"
EVENT = "StimOnset"  # event in behavior to align on
PRE_INTERVAL = 1000   # time in ms before event
POST_INTERVAL = 500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms


def get_feat_beh(session, feat):
    feat_beh = behavioral_utils.get_beh_model_labels_for_session_feat(session, feat, beh_path=SESS_BEHAVIOR_PATH)
    return feat_beh


def load_session_data(row, dim, condition):
    sess_name = row.session
    min_trials = row.MinTrialsPerCond

    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)
    valid_beh = behavioral_utils.get_valid_trials(beh)
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")
    beh = behavioral_utils.get_feature_values_per_session(sess_name, valid_beh)
    beh = behavioral_utils.get_max_feature_value(beh)
    def get_max_feat_chosen(row):
        dim = FEATURE_TO_DIM[row.MaxFeat]
        return row[dim] == row.MaxFeat
    beh["MaxFeatChosen"] = beh.apply(get_max_feat_chosen, axis=1)
    beh = beh[beh.MaxFeatChosen]

    dim_feats = POSSIBLE_FEATURES[dim]
    if condition == "attended":
        beh = beh[beh.MaxFeat.isin(dim_feats)]
    elif condition == "not_attended":
        beh = beh[~beh.MaxFeat.isin(dim_feats)]
    else: 
        raise ValueError("wrong condition")

    beh = beh.groupby(dim).sample(n=min_trials, random_state=DECODER_SEED)

    spikes_path = SESS_SPIKES_PATH.format(
        sess_name=sess_name, 
        pre_interval=PRE_INTERVAL, 
        event=EVENT, 
        post_interval=POST_INTERVAL, 
        interval_size=INTERVAL_SIZE
    )
    frs = pd.read_pickle(spikes_path)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    splitter = ConditionTrialSplitter(beh, dim, TEST_RATIO, seed=DECODER_SEED)
    session_data = SessionData(sess_name, beh, frs, splitter)
    session_data.pre_generate_splits(NUM_SPLITS)
    return session_data

def decode(dim_sesses, dim, condition):
    sess_datas = dim_sesses.apply(lambda row: load_session_data(row, dim, condition), axis=1)

    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = POSSIBLE_FEATURES[dim]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, NUM_SPLITS, NUM_TRAIN_PER_COND, NUM_TEST_PER_COND, DECODER_SEED
    ) 
    # store the results
    run_name = f"{dim}_{EVENT}_{condition}_"
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"{run_name}models.npy"), models)
    sess_datas.to_pickle(os.path.join(OUTPUT_DIR, f"{run_name}sess_datas.pickle"))

def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', default="Shape", type=str)
    parser.add_argument('--condition', default="attended")

    args = parser.parse_args()
    dim = args.feature_dim
    valid_sess = pd.read_pickle(SESSIONS_PATH)
    dim_sesses = valid_sess[valid_sess.dim == dim]
    print(f"Decoding between {dim} using {len(dim_sesses)} sessions")

    decode(dim_sesses, dim, args.condition)


if __name__ == "__main__":
    main()