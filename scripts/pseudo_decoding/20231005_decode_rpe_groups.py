import os
import numpy as np
import pandas as pd
import argparse
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils

from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 

EVENT = "FeedbackOnset"  # event in behavior to align on
PRE_INTERVAL = 1300   # time in ms before event
POST_INTERVAL = 1500  # time in ms after event
INTERVAL_SIZE = 100  # size of interval in ms

# the output directory to store the data
OUTPUT_DIR = "/data/patrick_res/pseudo"
# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/valid_sessions_rpe.pickle"
# path for each session, specifying behavior
SESS_BEHAVIOR_PATH = "/data/rawdata/sub-SA/sess-{sess_name}/behavior/sub-SA_sess-{sess_name}_object_features.csv"
# path for each session, for spikes that have been pre-aligned to event time and binned. 
SESS_SPIKES_PATH = "/data/patrick_res/firing_rates/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

# # the output directory to store the data
# OUTPUT_DIR = "/data/res/pseudo"
# # path to a dataframe of sessions to analyze
# # SESSIONS_PATH = "/data/patrick_scratch/multi_sess/valid_sessions.pickle"
# SESSIONS_PATH = "/data/valid_sessions_rpe.pickle"

# # path for each session, specifying behavior
# SESS_BEHAVIOR_PATH = "/data/sub-SA_sess-{sess_name}_object_features.csv"
# # path for each session, for spikes that have been pre-aligned to event time and binned. 
# SESS_SPIKES_PATH = "/data/{sess_name}_firing_rates_{pre_interval}_{event}_{post_interval}_{interval_size}_bins_1_smooth.pickle"

DATA_MODE = "SpikeCounts"

SEED = 42

RPE_GROUPS = ["more neg", "less neg", "less pos", "more pos"]


def load_session_data(row, subtrials): 
    """
    Loads the data (behavioral and firing rates) for a given session, 
    generates a TrialSplitter based on a condition (feature dimension) 
    creates a SessionData object to store data. 
    Args: 
        sess_name: name of the session to load
    Returns: a SessionData object
    """
    sess_name = row.session_name
    behavior_path = SESS_BEHAVIOR_PATH.format(sess_name=sess_name)
    beh = pd.read_csv(behavior_path)

    # filter trials 
    valid_beh = behavioral_utils.get_valid_trials(beh)

    # grab the features of the selected card
    feature_selections = behavioral_utils.get_selection_features(valid_beh)
    valid_beh = pd.merge(valid_beh, feature_selections, on="TrialNumber", how="inner")

    valid_beh_rpes = behavioral_utils.get_rpe_groups_per_session(sess_name, valid_beh)

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
    if subtrials is not None: 
        sess_subtrial = subtrials[subtrials.session == sess_name]
        valid_beh_rpes = valid_beh_rpes[valid_beh_rpes.TrialNumber.isin(sess_subtrial.TrialNumber)]
        if len(valid_beh_rpes) == 0: 
            return None
    # create a trial splitter 
    splitter = ConditionTrialSplitter(valid_beh_rpes, "RPEGroup", 0.2)
    return SessionData(sess_name, valid_beh_rpes, frs, splitter)


def run_decoder(sess_datas, subtrials_name="all", proj=None, proj_name="no_proj"):
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = None
    if subtrials_name == "all":
        classes = RPE_GROUPS
    elif subtrials_name == "cor_bal":
        classes = ["less pos", "more pos"]
    elif subtrials_name == "inc_bal":
        classes = ["less neg", "more neg"]

    # proj is in num_neurons x dimensions
    if proj is None:
        num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    else: 
        num_neurons = proj.shape[1]
    init_params = {"n_inputs": num_neurons, "p_dropout": 0.5, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(learning_rate=0.05, max_iter=500, batch_size=1000)
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)
    # calculate time bins (in seconds)
    time_bins = np.arange(0, (POST_INTERVAL + PRE_INTERVAL) / 1000, INTERVAL_SIZE / 1000)
    # train and evaluate the decoder per timein 
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, 10, 1000, 200, 42, proj)

    # store the results
    np.save(os.path.join(OUTPUT_DIR, f"rpe_groups_{proj_name}_{subtrials_name}_train_accs.npy"), train_accs)
    np.save(os.path.join(OUTPUT_DIR, f"rpe_groups_{proj_name}_{subtrials_name}_test_accs.npy"), test_accs)
    np.save(os.path.join(OUTPUT_DIR, f"rpe_groups_{proj_name}_{subtrials_name}_shuffled_accs.npy"), shuffled_accs)
    np.save(os.path.join(OUTPUT_DIR, f"rpe_groups_{proj_name}_{subtrials_name}_models.npy"), models)

def decode_rpe_group(valid_sess, subtrials=None, subtrials_name="all", proj=None, proj_name="no_proj"):
    print(f"Decoding")
    # load all session datas
    sess_datas = valid_sess.apply(lambda x: load_session_data(x, subtrials), axis=1)
    run_decoder(sess_datas, subtrials_name, proj, proj_name)


def main():
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtrials_path', type=str, help="a path to subtrials file", default="")
    parser.add_argument('--subtrials_name', type=str, help="name of subtrials", default="all")
    parser.add_argument('--proj_path', type=str, help="a path to projection file", default="")
    parser.add_argument('--proj_name', type=str, help="a path to projection file", default="no_proj")

    args = parser.parse_args()

    valid_sess = pd.read_pickle(SESSIONS_PATH)
    subtrials_name = args.subtrials_name
    proj_name = args.proj_name
    if args.subtrials_path:
        subtrials = pd.read_pickle(args.subtrials_path)
    else: 
        subtrials = None
    if args.proj_path:
        proj = np.load(args.proj_path)
    else: 
        proj = None
    decode_rpe_group(valid_sess, subtrials, subtrials_name, proj, proj_name)

if __name__ == "__main__":
    main()