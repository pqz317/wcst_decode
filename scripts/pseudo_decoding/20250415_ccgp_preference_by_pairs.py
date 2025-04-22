"""
Uses cross condition to evaluate the preference across different pairs
"""
import os
import numpy as np
import pandas as pd
import utils.pseudo_utils as pseudo_utils
import utils.pseudo_classifier_utils as pseudo_classifier_utils
import utils.behavioral_utils as behavioral_utils
import utils.spike_utils as spike_utils
import utils.io_utils as io_utils

from constants.behavioral_constants import *
from constants.decoding_constants import *
from utils.session_data import SessionData

from models.trainer import Trainer
from models.model_wrapper import ModelWrapper, ModelWrapperLinearRegression
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from trial_splitters.condition_trial_splitter import ConditionTrialSplitter 
from tqdm import tqdm
import argparse
from ccgp_preference_configs import add_defaults_to_parser

# path to a dataframe of sessions to analyze
SESSIONS_PATH = "/data/patrick_res/sessions/{sub}/valid_sessions.pickle"
SA_MORE_SESS_PAIRS_PATH = "/data/patrick_res/sessions/SA/pairs_at_least_3blocks_10sess_more_sess.pickle"
# BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_1blocks_3sess.pickle"
BL_PAIRS_PATH = "/data/patrick_res/sessions/BL/pairs_at_least_2blocks_6sess.pickle"

# path for each session, specifying behavior
# path for each session, for spikes that have been pre-aligned to event time and binned. 

DATA_MODE = "FiringRate"

UNITS_PATH = "/data/patrick_res/firing_rates/{sub}/all_units.pickle"

def load_session_data(row, feat, sub_units, args):
    """
    cond: either a feature or a pair of features: 
    TODO: refactor to match load_data in /src/wcst_decode/scripts/pseudo_decoding/decode_single_selected_features.py 
    """
    sess_name = row.session_name
    beh = behavioral_utils.load_behavior_from_args(sess_name, args)
    beh["FeatPreferred"] = beh["PreferredBelief"].apply(lambda x: "Preferred" if x == feat else "Not Preferred")
    if args.balance_by_filters: 
        beh = behavioral_utils.balance_trials_by_condition(beh, list(args.beh_filters.keys()))
    beh = behavioral_utils.filter_behavior(beh, args.beh_filters)

    # balance the conditions out: 
    sub_beh = behavioral_utils.balance_trials_by_condition(beh, ["FeatPreferred"])

    frs = io_utils.get_frs_from_args(args, sess_name)
    frs = frs.rename(columns={DATA_MODE: "Value"})
    if sub_units is not None: 
        frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
        frs = frs[frs.PseudoUnitID.isin(sub_units)]
    if len(frs) == 0 or len(sub_beh) == 0:
        return None
    splitter = ConditionTrialSplitter(sub_beh, "FeatPreferred", args.test_ratio)
    session_data = SessionData(sess_name, sub_beh, frs, splitter)
    session_data.pre_generate_splits(args.num_splits)
    return session_data

def load_sess_datas(args, cond, sub_units):
    sess_datas = args.sessions.apply(lambda row: load_session_data(row, cond, sub_units, args), axis=1)
    sess_datas = sess_datas.dropna()
    return sess_datas



def train_decoder(sess_datas, time_bins):
    # train the network
    # setup decoder, specify all possible label classes, number of neurons, parameters
    classes = ["Preferred", "Not Preferred"]
    num_neurons = sess_datas.apply(lambda x: x.get_num_neurons()).sum()
    print(f"Training with {len(sess_datas)} sessions, {num_neurons} units")
    init_params = {"n_inputs": num_neurons, "p_dropout": args.p_dropout, "n_classes": len(classes)}
    # create a trainer object
    trainer = Trainer(
        learning_rate=args.learning_rate, 
        max_iter=args.max_iter, 
        batch_size=1000
    )
    # create a wrapper for the decoder
    model = ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes)

    # calculate time bins (in seconds)
    train_accs, test_accs, shuffled_accs, models = pseudo_classifier_utils.evaluate_classifiers_by_time_bins(
        model, sess_datas, time_bins, args.num_splits, args.num_train_per_cond, args.num_test_per_cond
    ) 
    return train_accs, test_accs, shuffled_accs, models

    
def decode(args):
    sub_units = spike_utils.get_region_units(args.region_level, args.regions, UNITS_PATH.format(sub=args.subject))
    sub_units = spike_utils.get_sig_units(args, sub_units)
    trial_interval = args.trial_interval

    file_name = io_utils.get_ccgp_val_file_name(args)
    output_dir = io_utils.get_ccgp_val_output_dir(args)

    pair = args.feat_pair
    within_cond_accs = []
    across_cond_accs = []
    for feat in pair: 
        print(f"Training decoder preference in {feat}")
        # load up session data to train network
        train_feat_sess_datas = load_sess_datas(args, feat, sub_units)

        time_bins = np.arange(0, (trial_interval.post_interval + trial_interval.pre_interval) / 1000, trial_interval.interval_size / 1000)
        _, test_accs, _, models = train_decoder(train_feat_sess_datas, time_bins)
        within_cond_accs.append(test_accs)

        # next, evaluate network on other dimensions
        test_feat = [f for f in pair if f != feat][0]
        test_feat_sess_datas = load_sess_datas(args, test_feat, sub_units)
        accs_across_time = pseudo_classifier_utils.evaluate_model_with_data(models, test_feat_sess_datas, time_bins, num_test_per_cond=NUM_TEST_PER_COND)
        across_cond_accs.append(accs_across_time)
        np.save(os.path.join(output_dir, f"{file_name}_feat_{feat}_models.npy"), models)
    within_cond_accs = np.hstack(within_cond_accs)
    across_cond_accs = np.hstack(across_cond_accs)

    np.save(os.path.join(output_dir, f"{file_name}_within_cond_accs.npy"), within_cond_accs)
    np.save(os.path.join(output_dir, f"{file_name}_across_cond_accs.npy"), across_cond_accs)

def main(args):
    """
    Loads a dataframe specifying sessions to use
    For each feature dimension, runs decoding, stores results. 
    """
    subject = args.subject
    if subject == "SA": 
        pairs = pd.read_pickle(SA_MORE_SESS_PAIRS_PATH)
    else: 
        pairs = pd.read_pickle(BL_PAIRS_PATH)
    pair_row = pairs.iloc[args.pair_idx]
    args.feat_pair = pair_row.pair
    valid_sess = pd.read_pickle(SESSIONS_PATH.format(sub=subject))
    args.sessions = valid_sess[valid_sess.session_name.isin(pair_row.sessions)]
    args.trial_interval = get_trial_interval(args.trial_event)

    print(f"Computing CCGP for {subject} for preference for pairs {args.feat_pair} in {args.trial_event}", flush=True)
    print(f"shuffle idx is {args.shuffle_idx}", flush=True)
    print(f"Looking at regions {args.region_level}: {args.regions}, using use_next_trial_value {args.use_next_trial_value}", flush=True)
    print(f"examining conditions between {args.feat_pair} using between {len(args.sessions)} sessions", flush=True)
    print(f"Conditioning on prev response being {args.prev_response}", flush=True)
    decode(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_defaults_to_parser(parser)
    args = parser.parse_args()
    main(args)